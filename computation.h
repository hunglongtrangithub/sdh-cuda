#ifndef COMPUTATION
#define COMPUTATION

#include "kernels.h"
#include <stdint.h>

#define BOX_SIZE 23000

// Platform enum
enum platform { CPU, GPU };

// Function prototypes
int PDH_baseline(atoms_data *atoms, histogram *hist);
int time_and_fill_histogram_cpu(atoms_data *atoms, histogram *hist,
                                float *time);
int time_and_fill_histogram_gpu(atoms_data *atoms, histogram *hist,
                                uint64_t block_size, float *time,
                                enum kernel_algorithm algorithm);

#endif // !COMPUTATION
