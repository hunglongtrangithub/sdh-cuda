#ifndef EXPERIMENT
#define EXPERIMENT

#include "atoms.h"
#include "computation.h"
#include <stdint.h>

int experiment();

int run_gpu_version(atoms_data *atoms, histogram *cpu_hist, float time_cpu,
                    const char *version_name, enum kernel_algorithm gpu_ver,
                    double resolution, uint64_t block_size);

#endif // !EXPERIMENT
