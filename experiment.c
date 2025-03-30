#include "computation.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

int calculate_and_display_histogram(atoms_data *atoms, histogram *hist,
                                    enum platform platform, float *time,
                                    int count, ...) {
  switch (platform) {
  case CPU: {
    printf("Running CPU version\n");

    // Do the calculation and get the time
    if (time_and_fill_histogram_cpu(atoms, hist, time) != 0) {
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
    enum kernel_algorithm algorithm = (enum kernel_algorithm)algorithm_int;
    va_end(args);

    // Do the calculation and get the time
    if (time_and_fill_histogram_gpu(atoms, hist, block_size, time, algorithm) !=
        0) {
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
                    double resolution, unsigned long int block_size) {
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

void run_experiments(const char *filename) {
  // List of configurations to try
  size_t num_atoms_list[] = {10000, 50000, 100000};
  double resolution_list[] = {100, 200, 500};
  size_t block_sizes[] = {32, 64, 128, 256};

  // Initialize the CSV file with an improved header
  printf("Creating CSV file %s\n", filename);
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    perror("Failed to create CSV file");
    exit(EXIT_FAILURE);
  }
  fprintf(fp,
          "run_id,num_atoms,resolution,block_size,algorithm,time_ms,speedup\n");

  int run_id = 0;
  // Run a full cross-product of configurations
  for (size_t i = 0; i < sizeof(num_atoms_list) / sizeof(num_atoms_list[0]);
       i++) {
    for (size_t j = 0; j < sizeof(resolution_list) / sizeof(resolution_list[0]);
         j++) {
      for (size_t k = 0; k < sizeof(block_sizes) / sizeof(block_sizes[0]);
           k++) {
        size_t num_atoms = num_atoms_list[i];
        double resolution = resolution_list[j];
        size_t block_size = block_sizes[k];
        run_id++;

        printf("Running configuration %d: atoms=%zu, resolution=%f, "
               "block_size=%zu\n",
               run_id, num_atoms, resolution, block_size);

        // Initialize atoms
        double x_pos[num_atoms];
        double y_pos[num_atoms];
        double z_pos[num_atoms];

        atoms_data atoms = {x_pos, y_pos, z_pos, num_atoms};
        atoms_data_init(&atoms, BOX_SIZE);

        // Initialize the histogram
        size_t num_buckets = (size_t)(BOX_SIZE * sqrt(3) / resolution) + 1;
        bucket bucket[num_buckets];

        histogram hist = {
            .arr = bucket, .len = num_buckets, .resolution = resolution};
        histogram_init(&hist);

        // Measure CPU time
        float time_cpu = 0;
        if (time_and_fill_histogram_cpu(&atoms, &hist, &time_cpu) == 0) {
          fprintf(fp, "%d,%zu,%f,%zu,CPU,%.3f,1.0\n", run_id, num_atoms,
                  resolution, block_size, time_cpu);

          printf("  CPU time: %.3f ms\n", time_cpu);
        } else {
          printf("  CPU calculation failed\n");
        }

        // Reset histogram for GPU runs
        histogram_init(&hist);

        // Measure GPU times for different algorithms
        const char *gpu_algorithms[] = {"GRID_2D", "SHARED_MEM", "OUTPUT_PRIV"};
        enum kernel_algorithm algorithms[] = {GRID_2D, SHARED_MEMORY,
                                              OUTPUT_PRIVATIZATION};

        for (size_t alg = 0; alg < 3; alg++) {
          // Reset histogram for each algorithm
          histogram_init(&hist);

          float time_gpu = 0;
          if (time_and_fill_histogram_gpu(&atoms, &hist, block_size, &time_gpu,
                                          algorithms[alg]) == 0) {
            float speedup = time_cpu / time_gpu;

            fprintf(fp, "%d,%zu,%f,%zu,%s,%.3f,%.3f\n", run_id, num_atoms,
                    resolution, block_size, gpu_algorithms[alg], time_gpu,
                    speedup);

            printf("  %s time: %.3f ms (speedup: %.2fx)\n", gpu_algorithms[alg],
                   time_gpu, speedup);
          } else {
            printf("  %s calculation failed\n", gpu_algorithms[alg]);
          }
        }
      }
    }
  }
  fclose(fp);
}

int experiment() {
  // Create and initialize the CSV file
  const char *filename = "experiment_results.csv";

  // Run the experiments with predefined configurations
  printf("Running experiments...\n");
  run_experiments(filename);

  printf("Experiments completed. Results are saved in %s.\n", filename);
  return 0;
}
