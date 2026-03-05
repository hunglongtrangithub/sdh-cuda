#include <cstddef>
#include <math.h>
#include <stdint.h>

typedef struct {
  unsigned long long d_cnt;
} bucket;

__global__ void kernel_grid_2d(double *x_pos, double *y_pos, double *z_pos,
                               size_t atoms_len, bucket *hist, size_t hist_len,
                               double resolution) {
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= atoms_len || y >= atoms_len || x >= y)
    return;

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
  size_t h_pos = (size_t)(dist / resolution);
  if (h_pos >= hist_len)
    return;

  atomicAdd(&hist[h_pos].d_cnt, 1);
}

extern "C" void launch_grid_2d(double *x_pos, double *y_pos, double *z_pos,
                               size_t atoms_len, bucket *hist, size_t hist_len,
                               double resolution, uint32_t grid_dim_x,
                               uint32_t grid_dim_y, uint32_t grid_dim_z,
                               uint32_t block_dim_x, uint32_t block_dim_y,
                               uint32_t block_dim_z) {
  dim3 grid_dim(grid_dim_x, grid_dim_y, grid_dim_z);
  dim3 block_dim(block_dim_x, block_dim_y, block_dim_z);
  kernel_grid_2d<<<grid_dim, block_dim>>>(x_pos, y_pos, z_pos, atoms_len, hist,
                                          hist_len, resolution);
}
