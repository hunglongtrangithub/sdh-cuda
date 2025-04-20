#ifndef EXPERIMENT
#define EXPERIMENT

#include "atoms.h"
#include "computation.h"

int experiment();

int run_gpu_version(atoms_data *atoms, histogram *cpu_hist, float time_cpu,
                    const char *version_name, enum kernel_algorithm gpu_ver,
                    double resolution, unsigned long int block_size);

#endif // !EXPERIMENT
