#include "atom.h"
#include "computation.h"
#include "cuda_utils.h"
#include "histogram.h"
#include "kernels.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BOX_SIZE 23000

/* Results calculation and display function */
int calculate_and_display_histogram(atoms_data *atoms, histogram *hist,
                                    platform platform, float *time, int count,
                                    ...) {
  switch (platform) {
  case CPU: {
    printf("Running CPU version\n");

    // Do the calculation and get the time
    if (time_and_fill_histogram_cpu(atoms, hist, time) != 0) {
      fprintf(stderr, "Error running the algorithm on the CPU\n");
      return -1;
    }
    return 0;
  }
  case GPU: {
    printf("Running GPU version\n");

    va_list args;
    va_start(args, count);
    unsigned int block_size = va_arg(args, unsigned int);
    int algorithm_int = va_arg(args, int);
    kernel_algorithm algorithm = static_cast<kernel_algorithm>(algorithm_int);
    va_end(args);

    // Do the calculation and get the time
    if (time_and_fill_histogram_gpu(atoms, hist, block_size, time, algorithm) !=
        0) {
      fprintf(stderr, "Error running the algorithm on the GPU\n");
      return -1;
    }
    return 0;
  }
  default: {
    fprintf(stderr, "Unknown platform\n");
    return -1;
  }
  }
}

int run_gpu_version(atoms_data *atoms, histogram *cpu_hist, float time_cpu,
                    const char *version_name, enum kernel_algorithm gpu_ver,
                    double resolution, unsigned int block_size) {
  // Initialize histogram for this GPU version
  size_t num_buckets = cpu_hist->len;
  bucket buckets[num_buckets]; // Stack allocation
  histogram hist = {
      .arr = buckets, .len = num_buckets, .resolution = resolution};
  histogram_init(&hist);

  // Run the GPU version
  printf("========================================\n");
  printf("Running %s version\n", version_name);
  float time_gpu = 0;
  if (calculate_and_display_histogram(atoms, &hist, GPU, &time_gpu, 2,
                                      block_size, gpu_ver) != 0) {
    printf("Error running %s version. Exiting\n", version_name);
    return -1;
  }

  // Display histogram
  printf("%s version histogram:\n", version_name);
  display_histogram(&hist);
  printf("%s time in milliseconds: %f\n", version_name, time_gpu);

  // Calculate and display diff
  histogram diff_hist = hist; // Copy the structure
  for (size_t i = 0; i < diff_hist.len; i++) {
    diff_hist.arr[i].d_cnt -= cpu_hist->arr[i].d_cnt;
  }
  printf("%s version histogram diff:\n", version_name);
  display_histogram(&diff_hist);

  // Display speedup
  printf("Speedup (%s vs CPU): %f\n", version_name, time_cpu / time_gpu);
  printf("========================================\n");
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s {#of_samples} {bucket_width} {block_size}\n", argv[0]);
    return 1;
  }

  size_t particle_count = atoi(argv[1]);
  double resolution = atof(argv[2]);
  unsigned int block_size = atoi(argv[3]);

  // Initialize atoms on the stack
  double x_pos[particle_count];
  double y_pos[particle_count];
  double z_pos[particle_count];
  atoms_data atoms = {x_pos, y_pos, z_pos, particle_count};
  atoms_data_init(&atoms, BOX_SIZE);

  // The maximum distance between two points in a box is the diagonal
  size_t num_buckets = (BOX_SIZE * sqrt(3) / resolution) + 1;

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
