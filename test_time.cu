// write a simple vector addition kernel
// compile with: nvcc -o test_time test_time.cu
#include <stdio.h>
#define N 1000000

#define CHECK_CUDA_ERROR(call)                                                 \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      return -1;                                                               \
    }                                                                          \
  }

__global__ void add(int *a, int *b, int *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;
  printf("N = %d\n", N);
  for (int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c, N * sizeof(int)));

  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  add<<<N, 1>>>(dev_a, dev_b, dev_c);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(
      cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  CHECK_CUDA_ERROR(cudaFree(dev_a));
  CHECK_CUDA_ERROR(cudaFree(dev_b));
  CHECK_CUDA_ERROR(cudaFree(dev_c));
  printf("done\n");
  return 0;
}
