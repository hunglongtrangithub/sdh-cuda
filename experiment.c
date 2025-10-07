#include "computation.h"
#include "histogram.h"
#include "utils.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int run_gpu_version(atoms_data *atoms, histogram *cpu_hist, float time_cpu,
                    const char *version_name, enum kernel_algorithm gpu_ver,
                    double resolution, uint64_t block_size) {
  // Initialize histogram for this GPU version
  uint64_t num_buckets = cpu_hist->len;
  bucket *buckets = (bucket *)check_malloc(num_buckets * sizeof(bucket));

  histogram hist = {
      .arr = buckets, .len = num_buckets, .resolution = resolution};
  histogram_init(&hist);

  // Run the GPU version
  printf("========================================\n");
  printf("Running %s version\n", version_name);
  float time_gpu = 0;
  if (time_and_fill_histogram_gpu(atoms, &hist, block_size, &time_gpu,
                                  gpu_ver) != 0) {
    printf("Error running %s version. Exiting\n", version_name);
    return -1;
  }

  // Display histogram
  printf("%s version histogram:\n", version_name);
  display_histogram(&hist);
  printf("%s time in milliseconds: %f\n", version_name, time_gpu);

  // Calculate and display diff
  histogram diff_hist = hist; // Copy the structure
  for (uint64_t i = 0; i < diff_hist.len; i++) {
    diff_hist.arr[i].d_cnt -= cpu_hist->arr[i].d_cnt;
  }
  printf("%s version histogram diff:\n", version_name);
  display_histogram(&diff_hist);

  // Display speedup
  printf("Speedup (%s vs CPU): %f\n", version_name, time_cpu / time_gpu);
  printf("========================================\n");

  histogram_cleanup(&hist);
  return 0;
}

void run_experiments(const char *filename) {
  // List of configurations to try
  uint64_t num_atoms_list[] = {10000, 50000, 100000};
  double resolution_list[] = {100, 200, 500};
  uint64_t block_sizes[] = {32, 64, 128, 256};

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
  for (uint64_t i = 0; i < sizeof(num_atoms_list) / sizeof(num_atoms_list[0]);
       i++) {
    for (uint64_t j = 0;
         j < sizeof(resolution_list) / sizeof(resolution_list[0]); j++) {
      for (uint64_t k = 0; k < sizeof(block_sizes) / sizeof(block_sizes[0]);
           k++) {
        uint64_t num_atoms = num_atoms_list[i];
        double resolution = resolution_list[j];
        uint64_t block_size = block_sizes[k];
        run_id++;

        printf("Running configuration %d: atoms=%lu, resolution=%f, "
               "block_size=%lu\n",
               run_id, num_atoms, resolution, block_size);

        // Initialize atoms
        double x_pos[num_atoms];
        double y_pos[num_atoms];
        double z_pos[num_atoms];

        atoms_data atoms = {x_pos, y_pos, z_pos, num_atoms};
        atoms_data_init(&atoms, BOX_SIZE);

        // Initialize the histogram
        uint64_t num_buckets = (uint64_t)(BOX_SIZE * sqrt(3) / resolution) + 1;
        bucket bucket[num_buckets];

        histogram hist = {
            .arr = bucket, .len = num_buckets, .resolution = resolution};
        histogram_init(&hist);

        // Measure CPU time
        float time_cpu = 0;
        if (time_and_fill_histogram_cpu(&atoms, &hist, &time_cpu) == 0) {
          fprintf(fp, "%d,%lu,%f,%lu,CPU,%.3f,1.0\n", run_id, num_atoms,
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

        for (uint64_t alg = 0; alg < 3; alg++) {
          // Reset histogram for each algorithm
          histogram_init(&hist);

          float time_gpu = 0;
          if (time_and_fill_histogram_gpu(&atoms, &hist, block_size, &time_gpu,
                                          algorithms[alg]) == 0) {
            float speedup = time_cpu / time_gpu;

            fprintf(fp, "%d,%lu,%f,%lu,%s,%.3f,%.3f\n", run_id, num_atoms,
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
