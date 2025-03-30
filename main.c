#include "experiment.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  const char *help = "Usage:\n1. %s {#of_samples} {bucket_width} "
                     "{block_size}\n2. %s experiment\n";
  if (argc == 2) {
    if (strcmp(argv[1], "experiment") == 0) {
      experiment();
      return 1;
    } else {
      printf(help, argv[0], argv[0]);
      return 1;
    }
  }

  if (argc != 4) {
    printf(help, argv[0], argv[0]);
    return 1;
  }

  if (atol(argv[1]) <= 0) {
    printf("Invalid number of particles. Exiting\n");
    return 1;
  }
  size_t particle_count = (size_t)atol(argv[1]);

  double resolution = atof(argv[2]);

  if (atol(argv[3]) <= 0) {
    printf("Invalid block size. Exiting\n");
    return 1;
  }
  unsigned long int block_size = (unsigned long int)atol(argv[3]);

  // Initialize atoms on the stack
  double x_pos[particle_count];
  double y_pos[particle_count];
  double z_pos[particle_count];
  atoms_data atoms = {x_pos, y_pos, z_pos, particle_count};
  atoms_data_init(&atoms, BOX_SIZE);

  // The maximum distance between two points in a box is the diagonal
  size_t num_buckets = (size_t)(BOX_SIZE * sqrt(3) / resolution) + 1;

  // Run CPU version first to get reference histogram
  bucket buckets_cpu[num_buckets]; // Stack allocation
  histogram hist_cpu = {
      .arr = buckets_cpu,
      .len = num_buckets,
      .resolution = resolution,
  };
  histogram_init(&hist_cpu);

  float time_cpu = 0;
  if (calculate_and_display_histogram(&atoms, &hist_cpu, CPU, &time_cpu, 0, 0,
                                      0) != 0) {
    printf("Error running CPU version. Exiting\n");
    return 1;
  }

  // Store CPU time for speedup calculations

  printf("CPU version histogram:\n");
  display_histogram(&hist_cpu);
  printf("CPU time in milliseconds: %f\n", time_cpu);

  // Run GPU versions and compare to CPU
  if (run_gpu_version(&atoms, &hist_cpu, time_cpu, "GPU 2D grid (baseline)",
                      GRID_2D, resolution, block_size) != 0) {
    return 1;
  }
  if (run_gpu_version(&atoms, &hist_cpu, time_cpu, "GPU shared memory",
                      SHARED_MEMORY, resolution, block_size) != 0) {
    return 1;
  }
  if (run_gpu_version(&atoms, &hist_cpu, time_cpu, "GPU output privatization",
                      OUTPUT_PRIVATIZATION, resolution, block_size) != 0) {
    return 1;
  }

  return 0;
}
