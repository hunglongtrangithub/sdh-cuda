#ifndef KERNELS
#define KERNELS

#include "atom.h"
#include "histogram.h"

enum kernel_algorithm { GRID_2D, OUTPUT_PRIVATIZATION, SHARED_MEMORY };

int PDH_grid_2d(atoms_data *atoms_gpu, histogram *hist_gpu,
                unsigned long int block_size, float *time);
int PDH_shared_memory(atoms_data *atoms_gpu, histogram *hist_gpu,
                      unsigned long int block_size, float *time);
int PDH_output_privatization(atoms_data *atoms_gpu, histogram *hist_gpu,
                             unsigned long int block_size, float *time);

#endif // !KERNELS
