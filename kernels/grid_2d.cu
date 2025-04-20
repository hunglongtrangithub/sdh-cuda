#include "../atoms.h"
#include "../histogram.h"
#include "../utils.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

__global__ void kernel_grid_2d(double *x_pos, double *y_pos, double *z_pos,
                               uint64_t atoms_len, bucket *hist,
                               uint64_t hist_len, double resolution) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  // Filter out-of-bound threads and threads that have x >= y
  if (x >= atoms_len || y >= atoms_len || x >= y)
    return;

  // Calculate the distance between the two atoms
  double x1 = x_pos[x];
  double y1 = y_pos[x];
  double z1 = z_pos[x];

  double x2 = x_pos[y];
  double y2 = y_pos[y];
  double z2 = z_pos[y];

  double dx = x1 - x2;
  double dy = y1 - y2;
  double dz = z1 - z2;

  double dist = sqrt(dx * dx + dy * dy + dz * dz);
  // Calculate the histogram position
  int h_pos = (int)(dist / resolution);
  if (h_pos >= hist_len)
    return;

  // Increment the histogram count with atomic operation
  atomicAdd(&hist[h_pos].d_cnt, 1);
}

int PDH_grid_2d(atoms_data *atoms_gpu, histogram *hist_gpu,
                unsigned long int block_size, float *time) {
  // Check if CUDA device is available
  int device_count;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    fprintf(stderr, "No CUDA devices found\n");
    return -1;
  }

  cudaDeviceProp device_prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, 0));

  if ((int)block_size > device_prop.maxThreadsPerBlock) {
    fprintf(stderr, "Block size of %lu is too large. Must be less than %d\n",
            block_size, device_prop.maxThreadsPerBlock);
    return -1;
  }
  printf("Running baseline kernel using 2D grid\n");

  // Define the number of blocks and threads per block
  dim3 block_dim(sqrt(block_size), sqrt(block_size));
  dim3 grid_dim((atoms_gpu->len + block_dim.x - 1) / block_dim.x,
                (atoms_gpu->len + block_dim.y - 1) / block_dim.y);
  printf("Grid dimensions: %d x %d\n", grid_dim.x, grid_dim.y);
  printf("Block dimensions: %d x %d\n", block_dim.x, block_dim.y);

  cudaEvent_t start_time, end_time;
  CHECK_CUDA_ERROR(cudaEventCreate(&start_time));
  CHECK_CUDA_ERROR(cudaEventCreate(&end_time));

  // Start the timer
  CHECK_CUDA_ERROR(cudaEventRecord(start_time, 0));

  // Launch the kernel
  kernel_grid_2d<<<grid_dim, block_dim>>>(
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
