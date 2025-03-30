#include "../atom.h"
#include "../cuda_utils.h"
#include "../histogram.h"
#include <math.h>
#include <stdio.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

__global__ void
kernel_output_privatization(double *x_pos, double *y_pos, double *z_pos,
                            unsigned long long atoms_len, bucket *hist_2d,
                            unsigned int hist_len, double resolution) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // Shared memory layout: x/y/z plus block-private histogram
  extern __shared__ double shared_data[];
  double *x_shared = shared_data;
  double *y_shared = shared_data + blockDim.x;
  double *z_shared = shared_data + 2 * blockDim.x;

  // The privatized histogram for this block
  bucket *hist_shared = (bucket *)(shared_data + 3 * blockDim.x);

  // 1) Initialize the shared histogram to zero
  for (int bin = threadIdx.x; bin < (int)hist_len; bin += blockDim.x) {
    hist_shared[bin].d_cnt = 0ULL;
  }
  __syncthreads();

  // Local copy of the atom's position
  double x = 0.0, y = 0.0, z = 0.0;
  if (idx < atoms_len) {
    x = x_pos[idx];
    y = y_pos[idx];
    z = z_pos[idx];
  }
  __syncthreads();

  //------------------------------------------------------
  // 2) Inter-block comparisons
  //    Compare this block with each subsequent block block_id>blockIdx.x
  //------------------------------------------------------
  for (int block_id = blockIdx.x + 1; block_id < gridDim.x; block_id++) {
    // Each thread loads one atom from block block_id
    int i_global = block_id * blockDim.x + threadIdx.x;
    if (i_global < atoms_len) {
      x_shared[threadIdx.x] = x_pos[i_global];
      y_shared[threadIdx.x] = y_pos[i_global];
      z_shared[threadIdx.x] = z_pos[i_global];
    } else {
      // IMPORTANT: Zero out leftover threads so we don't read stale memory
      x_shared[threadIdx.x] = 0.0;
      y_shared[threadIdx.x] = 0.0;
      z_shared[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // Number of valid atoms in that block
    int valid_count =
        min((int)blockDim.x,
            (int)(atoms_len - (unsigned long long)block_id * blockDim.x));
    if (valid_count < 0)
      valid_count = 0; // clamp

    // Compare our local atom with each valid atom in block block_id
    for (int j = 0; j < valid_count; j++) {
      double dx = x - x_shared[j];
      double dy = y - y_shared[j];
      double dz = z - z_shared[j];
      double dist = sqrt(dx * dx + dy * dy + dz * dz);

      int h_pos = (int)(dist / resolution);
      if (h_pos >= 0 && h_pos < (int)hist_len) {
        atomicAdd(&hist_shared[h_pos].d_cnt, 1ULL);
      }
    }
    __syncthreads();
  }

  //------------------------------------------------------
  // 3) Intra-block comparisons: compare atoms within the same block
  //------------------------------------------------------
  // Reload *this* block's atom into shared memory
  if (idx < atoms_len) {
    x_shared[threadIdx.x] = x;
    y_shared[threadIdx.x] = y;
    z_shared[threadIdx.x] = z;
  } else {
    x_shared[threadIdx.x] = 0.0;
    y_shared[threadIdx.x] = 0.0;
    z_shared[threadIdx.x] = 0.0;
  }
  __syncthreads();

  // Number of valid atoms in this block
  int leftover =
      min((int)blockDim.x,
          (int)(atoms_len - (unsigned long long)blockIdx.x * blockDim.x));
  if (leftover < 0)
    leftover = 0;

  // Each thread compares its atom with atoms at higher thread indices
  for (int i = threadIdx.x + 1; i < leftover; i++) {
    double dx = x - x_shared[i];
    double dy = y - y_shared[i];
    double dz = z - z_shared[i];
    double dist = sqrt(dx * dx + dy * dy + dz * dz);

    int h_pos = (int)(dist / resolution);
    if (h_pos >= 0 && h_pos < (int)hist_len) {
      atomicAdd(&hist_shared[h_pos].d_cnt, 1ULL);
    }
  }
  __syncthreads();

  //------------------------------------------------------
  // 4) Write the block's privatized histogram to global memory
  //------------------------------------------------------
  // Row = bin in [0..hist_len), Column = blockIdx.x
  for (int bin = threadIdx.x; bin < (int)hist_len; bin += blockDim.x) {
    hist_2d[bin * gridDim.x + blockIdx.x].d_cnt = hist_shared[bin].d_cnt;
  }
}

__global__ void kernel_reduction(bucket *hist_2d, int hist_2d_width,
                                 bucket *hist) {
  // Load to shared memory
  extern __shared__ bucket shared_mem[];
  shared_mem[threadIdx.x].d_cnt =
      threadIdx.x < hist_2d_width
          ? hist_2d[blockIdx.x * hist_2d_width + threadIdx.x].d_cnt
          : 0;
  __syncthreads();
  // Reduce
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_mem[threadIdx.x].d_cnt += shared_mem[threadIdx.x + stride].d_cnt;
    }
    __syncthreads();
  }
  // Write to global memory
  if (threadIdx.x == 0) {
    hist[blockIdx.x].d_cnt = shared_mem[0].d_cnt;
  }
}

int PDH_output_privatization(atoms_data *atoms_gpu, histogram *hist_gpu,
                             unsigned long int block_size, float *time) {
  // Check if CUDA device is available
  int device_count;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    fprintf(stderr, "No CUDA devices found\n");
    return -1;
  }

  cudaDeviceProp deviceProp;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));

  // Launch the kernel
  printf("Launching shared memory kernel\n");

  int grid_size = (atoms_gpu->len + block_size - 1) / block_size;
  // We need to allocate enough space for the block size * 3 (x, y, z) and
  // histogram
  size_t shared_mem_size =
      3 * block_size * sizeof(double) + hist_gpu->len * sizeof(bucket);
  if (shared_mem_size > deviceProp.sharedMemPerBlock) {
    fprintf(stderr,
            "Shared memory size of %zu is too large. Must be less than %zu\n",
            shared_mem_size, deviceProp.sharedMemPerBlock);
    return -1;
  }

  printf("Grid size: %d\n", grid_size);
  printf("Block size: %lu\n", block_size);
  printf("Shared memory size: %zu\n", shared_mem_size);

  // Define the histogram 2D grid for all blocks
  bucket *hist_2d;
  CHECK_CUDA_ERROR(
      cudaMalloc(&hist_2d, hist_gpu->len * grid_size * sizeof(bucket)));

  cudaEvent_t start_time, end_time;
  CHECK_CUDA_ERROR(cudaEventCreate(&start_time));
  CHECK_CUDA_ERROR(cudaEventCreate(&end_time));

  // Start the timer
  CHECK_CUDA_ERROR(cudaEventRecord(start_time, 0));
  kernel_output_privatization<<<grid_size, block_size, shared_mem_size>>>(
      atoms_gpu->x_pos, atoms_gpu->y_pos, atoms_gpu->z_pos, atoms_gpu->len,
      hist_2d, hist_gpu->len, hist_gpu->resolution);
  CHECK_CUDA_ERROR(cudaPeekAtLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Launch the reduction kernel
  printf("Launching reduction kernel\n");

  // Round up to the next power of 2 for the block size to do the reduction
  int reduction_block_size = pow(2, ceil(log2(grid_size + 1)));
  if (reduction_block_size > deviceProp.maxThreadsPerBlock) {
    fprintf(stderr, "Block size of %d is too large. Must be less than %d\n",
            reduction_block_size, deviceProp.maxThreadsPerBlock);
    CHECK_CUDA_ERROR(cudaFree(hist_2d));
    return -1;
  }

  size_t reduction_shared_mem_size = reduction_block_size * sizeof(bucket);
  if (reduction_shared_mem_size > deviceProp.sharedMemPerBlock) {
    fprintf(stderr,
            "Shared memory size of %zu is too large. Must be less than %zu\n",
            reduction_shared_mem_size, deviceProp.sharedMemPerBlock);
    CHECK_CUDA_ERROR(cudaFree(hist_2d));
    return -1;
  }

  printf("Reduction grid size: %zu\n", hist_gpu->len);
  printf("Reduction block size: %d\n", reduction_block_size);
  printf("Reduction shared memory size: %zu\n", reduction_shared_mem_size);

  // Launch the reduction kernel
  kernel_reduction<<<hist_gpu->len, reduction_block_size,
                     reduction_shared_mem_size>>>(hist_2d, grid_size,
                                                  hist_gpu->arr);
  CHECK_CUDA_ERROR(cudaPeekAtLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Stop the timer
  CHECK_CUDA_ERROR(cudaEventRecord(end_time, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(end_time));

  // Free the histogram 2D grid
  CHECK_CUDA_ERROR(cudaFree(hist_2d));

  // Calculate the elapsed time
  CHECK_CUDA_ERROR(cudaEventElapsedTime(time, start_time, end_time));
  CHECK_CUDA_ERROR(cudaEventDestroy(start_time));
  CHECK_CUDA_ERROR(cudaEventDestroy(end_time));

  return 0;
}
