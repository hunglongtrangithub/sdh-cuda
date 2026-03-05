#include <cstddef>
#include <cstdint>
#include <math.h>
#include <stdint.h>

typedef struct {
  unsigned long long d_cnt;
} bucket;

#define min(a, b) ((a) < (b) ? (a) : (b))

__global__ void kernel_output_privatization(double *x_pos, double *y_pos,
                                            double *z_pos, size_t atoms_len,
                                            bucket *hist_2d, size_t hist_len,
                                            double resolution) {
  size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

  // Shared memory layout: x/y/z plus block-private histogram
  extern __shared__ double shared_data[];
  double *x_shared = shared_data;
  double *y_shared = shared_data + blockDim.x;
  double *z_shared = shared_data + 2 * blockDim.x;
  bucket *hist_shared = (bucket *)(shared_data + 3 * blockDim.x);

  // 1) Initialize the shared histogram to zero
  for (size_t bin = threadIdx.x; bin < hist_len; bin += blockDim.x) {
    hist_shared[bin].d_cnt = 0;
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

  // 2) Inter-block comparisons
  //    Compare this block with each subsequent block block_id>blockIdx.x
  for (size_t block_id = blockIdx.x + 1; block_id < gridDim.x; block_id++) {
    size_t i_global = block_id * blockDim.x + threadIdx.x;
    if (i_global < atoms_len) {
      x_shared[threadIdx.x] = x_pos[i_global];
      y_shared[threadIdx.x] = y_pos[i_global];
      z_shared[threadIdx.x] = z_pos[i_global];
    }
    __syncthreads();

    size_t valid_count =
        min((size_t)blockDim.x, atoms_len - block_id * blockDim.x);

    for (size_t j = 0; j < valid_count; j++) {
      double dx = x - x_shared[j];
      double dy = y - y_shared[j];
      double dz = z - z_shared[j];
      double dist = sqrt(dx * dx + dy * dy + dz * dz);

      size_t h_pos = (size_t)(dist / resolution);
      if (h_pos < hist_len) {
        atomicAdd(&hist_shared[h_pos].d_cnt, 1ULL);
      }
    }
    __syncthreads();
  }

  // 3) Intra-block comparisons: compare atoms within the same block
  if (idx < atoms_len) {
    x_shared[threadIdx.x] = x;
    y_shared[threadIdx.x] = y;
    z_shared[threadIdx.x] = z;
  }
  __syncthreads();

  size_t valid_count =
      min((size_t)blockDim.x, atoms_len - (size_t)blockIdx.x * blockDim.x);

  for (size_t i = threadIdx.x + 1; i < valid_count; i++) {
    double dx = x - x_shared[i];
    double dy = y - y_shared[i];
    double dz = z - z_shared[i];
    double dist = sqrt(dx * dx + dy * dy + dz * dz);

    size_t h_pos = (size_t)(dist / resolution);
    if (h_pos < (int)hist_len) {
      atomicAdd(&hist_shared[h_pos].d_cnt, 1);
    }
  }
  __syncthreads();

  // 4) Write the block's privatized histogram to global memory
  // hist_2d is a 2D array stored in row-major order. Its dimensions are
  // (hist_len, gridDim.x). Threads in the block cooperatively write the
  // privatized histogram to the appropriate column (blockIdx.x), in strides of
  // blockDim.x.
  // Row index = bin in [0..hist_len), Column index = blockIdx.x
  for (size_t bin = threadIdx.x; bin < hist_len; bin += blockDim.x) {
    hist_2d[bin * gridDim.x + blockIdx.x].d_cnt = hist_shared[bin].d_cnt;
  }
}

__global__ void kernel_reduction(bucket *hist_2d, size_t hist_2d_width,
                                 bucket *hist) {
  extern __shared__ bucket shared_mem[];

  shared_mem[threadIdx.x].d_cnt = 0;

  // Each thread handles multiple elements when hist_2d_width > blockDim.x
  for (size_t i = threadIdx.x; i < hist_2d_width; i += blockDim.x) {
    shared_mem[threadIdx.x].d_cnt +=
        hist_2d[blockIdx.x * hist_2d_width + i].d_cnt;
  }
  __syncthreads();

  // Reduce within the block
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_mem[threadIdx.x].d_cnt += shared_mem[threadIdx.x + stride].d_cnt;
    }
    __syncthreads();
  }

  // Write result to global memory
  if (threadIdx.x == 0) {
    hist[blockIdx.x].d_cnt = shared_mem[0].d_cnt;
  }
}

extern "C" void
launch_output_privatization(double *x_pos, double *y_pos, double *z_pos,
                            size_t atoms_len, bucket *hist_2d, size_t hist_len,
                            double resolution, uint32_t grid_size,
                            uint32_t block_size, size_t shared_mem_size) {
  kernel_output_privatization<<<grid_size, block_size, shared_mem_size>>>(
      x_pos, y_pos, z_pos, atoms_len, hist_2d, hist_len, resolution);
}

extern "C" void launch_reduction(bucket *hist_2d, size_t hist_2d_width,
                                 bucket *hist, uint32_t reduction_grid_size,
                                 uint32_t reduction_block_size,
                                 size_t reduction_shared_mem_size) {
  kernel_reduction<<<reduction_grid_size, reduction_block_size,
                     reduction_shared_mem_size>>>(hist_2d, hist_2d_width, hist);
}
