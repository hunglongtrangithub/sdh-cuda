#include <math.h>
#include <stdint.h>

typedef struct {
  unsigned long long d_cnt;
} bucket;

#define min(a, b) ((a) < (b) ? (a) : (b))

__global__ void kernel_shared_memory(double *x_pos, double *y_pos,
                                     double *z_pos, size_t atoms_len,
                                     bucket *hist, size_t hist_len,
                                     double resolution) {
  size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= atoms_len)
    return;

  extern __shared__ double shared_data[];
  double *x_shared = shared_data;
  double *y_shared = shared_data + blockDim.x;
  double *z_shared = shared_data + 2 * blockDim.x;

  double x = x_pos[idx];
  double y = y_pos[idx];
  double z = z_pos[idx];

  // Inter-block comparisons
  for (int block_id = blockIdx.x + 1; block_id < gridDim.x; block_id++) {
    int i = blockDim.x * block_id + threadIdx.x;
    if (i < atoms_len) {
      x_shared[threadIdx.x] = x_pos[i];
      y_shared[threadIdx.x] = y_pos[i];
      z_shared[threadIdx.x] = z_pos[i];
    }
    __syncthreads();

    for (size_t j = 0; j < min(blockDim.x, atoms_len - block_id * blockDim.x);
         j++) {
      double dist = sqrt((x - x_shared[j]) * (x - x_shared[j]) +
                         (y - y_shared[j]) * (y - y_shared[j]) +
                         (z - z_shared[j]) * (z - z_shared[j]));
      size_t h_pos = (size_t)(dist / resolution);
      if (h_pos < hist_len) {
        atomicAdd(&hist[h_pos].d_cnt, 1);
      }
    }
    __syncthreads();
  }

  // Intra-block comparisons
  x_shared[threadIdx.x] = x;
  y_shared[threadIdx.x] = y;
  z_shared[threadIdx.x] = z;
  __syncthreads();

  for (size_t i = threadIdx.x + 1;
       i < min(blockDim.x, atoms_len - blockIdx.x * blockDim.x); i++) {
    double dist = sqrt((x - x_shared[i]) * (x - x_shared[i]) +
                       (y - y_shared[i]) * (y - y_shared[i]) +
                       (z - z_shared[i]) * (z - z_shared[i]));
    size_t h_pos = (size_t)(dist / resolution);
    if (h_pos < hist_len) {
      atomicAdd(&hist[h_pos].d_cnt, 1);
    }
  }
}

extern "C" void launch_shared_memory(double *x_pos, double *y_pos,
                                     double *z_pos, size_t atoms_len,
                                     bucket *hist, size_t hist_len,
                                     double resolution, uint32_t grid_size,
                                     uint32_t block_size,
                                     size_t shared_mem_size) {
  kernel_shared_memory<<<grid_size, block_size, shared_mem_size>>>(
      x_pos, y_pos, z_pos, atoms_len, hist, hist_len, resolution);
}
