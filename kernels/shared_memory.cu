#include "../atom.h"
#include "../cuda_utils.h"
#include "../histogram.h"
#include <stdio.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

__global__ void kernel_shared_memory(double *x_pos, double *y_pos,
                                     double *z_pos,
                                     unsigned long long atoms_len, bucket *hist,
                                     unsigned int hist_len, double resolution) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= atoms_len)
    return;
  extern __shared__ double shared_data[]; // Single shared memory block
  double *x_shared = shared_data;
  double *y_shared = shared_data + blockDim.x;
  double *z_shared = shared_data + 2 * blockDim.x;

  double x = x_pos[idx];
  double y = y_pos[idx];
  double z = z_pos[idx];

  // Loop through each next block from the current block
  for (int block_id = blockIdx.x + 1; block_id < gridDim.x; block_id++) {
    // Load the block into shared memory
    int i = blockDim.x * block_id + threadIdx.x;
    if (i < atoms_len) {
      x_shared[threadIdx.x] = x_pos[i];
      y_shared[threadIdx.x] = y_pos[i];
      z_shared[threadIdx.x] = z_pos[i];
    }
    __syncthreads();

    // Loop through each atom in the cached block
    for (int j = 0; j < min(blockDim.x, atoms_len - block_id * blockDim.x);
         j++) {
      double dist = sqrt((x - x_shared[j]) * (x - x_shared[j]) +
                         (y - y_shared[j]) * (y - y_shared[j]) +
                         (z - z_shared[j]) * (z - z_shared[j]));
      int h_pos = (int)(dist / resolution);
      if (h_pos < hist_len) {
        atomicAdd(&hist[h_pos].d_cnt, 1);
      }
    }
    __syncthreads();
  }

  // Load the current block into shared memory
  x_shared[threadIdx.x] = x;
  y_shared[threadIdx.x] = y;
  z_shared[threadIdx.x] = z;
  __syncthreads();

  // Loop through each atom in the current block
  for (int i = threadIdx.x + 1;
       i < min(blockDim.x, atoms_len - blockIdx.x * blockDim.x); i++) {
    double dist = sqrt((x - x_shared[i]) * (x - x_shared[i]) +
                       (y - y_shared[i]) * (y - y_shared[i]) +
                       (z - z_shared[i]) * (z - z_shared[i]));
    int h_pos = (int)(dist / resolution);
    if (h_pos < hist_len) {
      atomicAdd(&hist[h_pos].d_cnt, 1);
    }
  }
}

int PDH_shared_memory(atoms_data *atoms_gpu, histogram *hist_gpu,
                      unsigned long int block_size, float *time) {
  printf("Running kernel using shared memory\n");

  // Check if CUDA device is available
  int device_count;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    fprintf(stderr, "No CUDA devices found\n");
    return -1;
  }

  cudaDeviceProp device_prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, 0));

  int grid_size = (atoms_gpu->len + block_size - 1) / block_size;
  // We need to allocate enough space for the block size * 3 (x, y, z)
  size_t shared_mem_size = 3 * block_size * sizeof(double);
  if (shared_mem_size > device_prop.sharedMemPerBlock) {
    fprintf(stderr, "Shared memory size is too large. Must be less than %zu\n",
            device_prop.sharedMemPerBlock);
    return -1;
  }

  printf("Grid size: %d\n", grid_size);
  printf("Block size: %lu\n", block_size);
  printf("Shared memory size: %zu\n", shared_mem_size);

  cudaEvent_t start_time, end_time;
  CHECK_CUDA_ERROR(cudaEventCreate(&start_time));
  CHECK_CUDA_ERROR(cudaEventCreate(&end_time));

  // Start the timer
  CHECK_CUDA_ERROR(cudaEventRecord(start_time, 0));

  kernel_shared_memory<<<grid_size, block_size, shared_mem_size>>>(
      atoms_gpu->x_pos, atoms_gpu->y_pos, atoms_gpu->z_pos, atoms_gpu->len,
      hist_gpu->arr, hist_gpu->len, hist_gpu->resolution);
  CHECK_CUDA_ERROR(cudaPeekAtLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Stop the timer
  CHECK_CUDA_ERROR(cudaEventRecord(end_time, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(end_time));

  // Calculate the elapsed time
  CHECK_CUDA_ERROR(cudaEventElapsedTime(time, start_time, end_time));
  CHECK_CUDA_ERROR(cudaEventDestroy(start_time));
  CHECK_CUDA_ERROR(cudaEventDestroy(end_time));

  return 0;
}
