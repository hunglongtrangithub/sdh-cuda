#ifndef COMPUTATION
#define COMPUTATION

#include "atom.h"
#include "histogram.h"
#include "kernels.h"

// Platform enum
enum platform { CPU, GPU };

// Function prototypes
int PDH_baseline(atoms_data *atoms, histogram *hist);
int time_and_fill_histogram_cpu(atoms_data *atoms, histogram *hist,
                                float *time);
int time_and_fill_histogram_gpu(atoms_data *atoms, histogram *hist,
                                unsigned int block_size, float *time,
                                kernel_algorithm algorithm);

#endif // !COMPUTATION
