#include "atom.h"
#include "computation.h"
#include "histogram.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to initialize the CSV file and write the header
void init_csv_file(const char *filename) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    perror("Failed to create CSV file");
    exit(EXIT_FAILURE);
  }
  fprintf(fp, "num_atoms,resolution,block_size,algorithm,time_ms\n");
  fclose(fp);
}

// Function to append results to the CSV file
void append_to_csv(const char *filename, size_t num_atoms, double resolution,
                   unsigned int block_size, const char *algorithm,
                   float time_ms) {
  FILE *fp = fopen(filename, "a");
  if (!fp) {
    perror("Failed to open CSV file");
    return;
  }
  fprintf(fp, "%zu,%.2f,%u,%s,%.3f\n", num_atoms, resolution, block_size,
          algorithm, time_ms);
  fclose(fp);
}

// Function to run the experiments
void run_experiments(const char *filename) {
  // List of configurations to try
  size_t num_atoms_list[] = {10000, 50000, 100000};
  double resolution_list[] = {0.1, 0.2, 0.5};
  size_t block_sizes[] = {32, 64, 128, 256};

  // Loop through different atom counts, resolutions, and block sizes
  for (size_t i = 0; i < sizeof(num_atoms_list) / sizeof(num_atoms_list[0]);
       i++) {
    size_t num_atoms = num_atoms_list[i];
    double resolution = resolution_list[i % (sizeof(resolution_list) /
                                             sizeof(resolution_list[0]))];
    unsigned int block_size =
        block_sizes[i % (sizeof(block_sizes) / sizeof(block_sizes[0]))];

    // Initialize atoms
    double x_pos[num_atoms];
    double y_pos[num_atoms];
    double z_pos[num_atoms];
    atoms_data atoms = {x_pos, y_pos, z_pos, num_atoms};
    atoms_data_init(&atoms, BOX_SIZE);

    // Initialize the histogram
    size_t num_buckets = (BOX_SIZE * sqrt(3) / resolution) + 1;
    bucket buckets[num_buckets];
    histogram hist = {
        .arr = buckets, .len = num_buckets, .resolution = resolution};
    histogram_init(&hist);

    // Measure CPU time
    float time_cpu = 0;
    if (time_and_fill_histogram_cpu(&atoms, &hist, &time_cpu) == 0) {
      append_to_csv(filename, num_atoms, resolution, block_size, "CPU",
                    time_cpu);
    }

    // Measure GPU times for different algorithms
    const char *gpu_algorithms[] = {"GPU 2D grid", "GPU shared memory",
                                    "GPU output privatization"};
    enum kernel_algorithm algorithms[] = {GRID_2D, SHARED_MEMORY,
                                          OUTPUT_PRIVATIZATION};

    for (size_t j = 0; j < 3; j++) {
      float time_gpu = 0;
      if (time_and_fill_histogram_gpu(&atoms, &hist, block_size, &time_gpu,
                                      algorithms[j]) == 0) {
        append_to_csv(filename, num_atoms, resolution, block_size,
                      gpu_algorithms[j], time_gpu);
      }
    }
  }
}

int main() {
  // Create and initialize the CSV file
  const char *filename = "experiment_results.csv";
  init_csv_file(filename);

  // Run the experiments with predefined configurations
  run_experiments(filename);

  printf("Experiments completed. Results are saved in %s.\n", filename);
  return 0;
}
