#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));            \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

__global__ void reduceSum(int *input, int *output, int n) {
  extern __shared__ int sharedData[];

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sharedData[tid] = (idx < n) ? input[idx] : 0;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sharedData[tid] += sharedData[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sharedData[0];
  }
}

int main(int argc, char *argv[]) {
  int n = 1024;
  if (argc > 1) {
    n = atoi(argv[1]);
  }

  printf("Array size: %d\n", n);

  int *h_input = (int *)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    h_input[i] = 1;
  }

  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  int *d_input, *d_output;
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, n * sizeof(int)));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

  int sharedMemSize = threadsPerBlock * sizeof(int);

  int remaining = n;
  int *input_ptr = d_input;

  while (remaining > 1) {
    numBlocks = (remaining + threadsPerBlock - 1) / threadsPerBlock;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, numBlocks * sizeof(int)));

    reduceSum<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        input_ptr, d_output, remaining);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (input_ptr != d_input) {
      cudaFree(input_ptr);
    }

    input_ptr = d_output;
    remaining = numBlocks;
  }

  int h_sum;
  CHECK_CUDA_ERROR(
      cudaMemcpy(&h_sum, input_ptr, sizeof(int), cudaMemcpyDeviceToHost));

  printf("Sum: %d\n", h_sum);
  printf("Expected sum: %d\n", n);
  printf("Result %s\n", (h_sum == n) ? "CORRECT" : "INCORRECT");

  CHECK_CUDA_ERROR(cudaFree(input_ptr));
  CHECK_CUDA_ERROR(cudaFree(d_input));
  free(h_input);

  return 0;
}
