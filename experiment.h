#ifndef EXPERIMENT
#define EXPERIMENT

#include "atom.h"
#include "computation.h"

int experiment();

int calculate_and_display_histogram(atoms_data *atoms, histogram *hist,
                                    enum platform platform, float *time,
                                    int count, ...);

int run_gpu_version(atoms_data *atoms, histogram *cpu_hist, float time_cpu,
                    const char *version_name, enum kernel_algorithm gpu_ver,
                    double resolution, unsigned long int block_size);

#endif // !EXPERIMENT
