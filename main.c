#include "experiment.h"
#include "utils.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int demo(uint64_t particle_count, double resolution, uint64_t block_size) {
  // Allocate memory for atoms
  double *x_pos = (double *)check_malloc(particle_count * sizeof(double));
  double *y_pos = (double *)check_malloc(particle_count * sizeof(double));
  double *z_pos = (double *)check_malloc(particle_count * sizeof(double));

  atoms_data atoms = {x_pos, y_pos, z_pos, particle_count};
  atoms_data_init(&atoms, BOX_SIZE);

  // The maximum distance between two points in a box is the diagonal
  uint64_t num_buckets = (uint64_t)(BOX_SIZE * sqrt(3) / resolution) + 1;

  // Run CPU version first to get reference histogram
  bucket *buckets_cpu = (bucket *)check_malloc(num_buckets * sizeof(bucket));

  histogram hist_cpu = {
      .arr = buckets_cpu, .len = num_buckets, .resolution = resolution};
  histogram_init(&hist_cpu);

  float time_cpu;
  printf("Running CPU version...\n");
  if (time_and_fill_histogram_cpu(&atoms, &hist_cpu, &time_cpu) != 0) {
    fprintf(stderr, "CPU histogram computation failed\n");
    atoms_data_cleanup(&atoms);
    histogram_cleanup(&hist_cpu);
    return 1;
  }
  printf("CPU time in milliseconds: %f\n", time_cpu);

  // Run GPU versions and compare to CPU
  int result = 0;
  result |=
      run_gpu_version(&atoms, &hist_cpu, time_cpu, "GPU 2D grid (baseline)",
                      GRID_2D, resolution, block_size);
  result |= run_gpu_version(&atoms, &hist_cpu, time_cpu, "GPU shared memory",
                            SHARED_MEMORY, resolution, block_size);
  result |=
      run_gpu_version(&atoms, &hist_cpu, time_cpu, "GPU output privatization",
                      OUTPUT_PRIVATIZATION, resolution, block_size);

  // Cleanup
  atoms_data_cleanup(&atoms);
  histogram_cleanup(&hist_cpu);

  return result;
}

int main(int argc, char **argv) {
  const char *help = "Usage:\n1. %s {#of_samples} {bucket_width} "
                     "{block_size}\n2. %s experiment\n";
  if (argc == 2) {
    if (strcmp(argv[1], "experiment") == 0) {
      experiment();
      return 0;
    } else {
      printf(help, argv[0], argv[0]);
      return 1;
    }
  }

  if (argc != 4) {
    printf(help, argv[0], argv[0]);
    return 1;
  }

  if (atoll(argv[1]) <= 0) {
    printf("Invalid number of particles. Exiting\n");
    return 1;
  }
  uint64_t particle_count = (uint64_t)atoll(argv[1]);

  double resolution = atof(argv[2]);

  if (atoll(argv[3]) <= 0) {
    printf("Invalid block size. Exiting\n");
    return 1;
  }
  uint64_t block_size = (uint64_t)atoll(argv[3]);

  return demo(particle_count, resolution, block_size);
}
