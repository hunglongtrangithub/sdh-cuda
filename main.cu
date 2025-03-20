#include "atom.h"
#include "cuda_utils.h"
#include "histogram.h"
#include "kernels.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BOX_SIZE 23000
enum platform { CPU, GPU };

int PDH_baseline(atoms_data *atoms, histogram *hist) {
  size_t i, j;
  double dist;

  for (i = 0; i < atoms->len; i++) {
    for (j = i + 1; j < atoms->len; j++) {
      double x1 = atoms->x_pos[i];
      double y1 = atoms->y_pos[i];
      double z1 = atoms->z_pos[i];

      double x2 = atoms->x_pos[j];
      double y2 = atoms->y_pos[j];
      double z2 = atoms->z_pos[j];

      double dx = x1 - x2;
      double dy = y1 - y2;
      double dz = z1 - z2;

      dist = sqrt(dx * dx + dy * dy + dz * dz);
      size_t h_pos = dist / hist->resolution;
      if (h_pos >= hist->len)
        continue;
      hist->arr[h_pos].d_cnt++;
    }
  }

  return 0;
}

int time_and_fill_histogram_cpu(atoms_data *atoms, histogram *hist,
                                float *time) {
  struct timespec start;
  struct timespec end;

  // Record the start time
  clock_gettime(CLOCK_MONOTONIC, &start);

  if (PDH_baseline(atoms, hist) != 0) {
    return -1;
  }

  // Record the end time
  clock_gettime(CLOCK_MONOTONIC, &end);

  // Calculate the time difference
  struct timespec time_diff = {.tv_sec = end.tv_sec - start.tv_sec,
                               .tv_nsec = end.tv_nsec - start.tv_nsec};
  if (time_diff.tv_nsec < 0) {
    time_diff.tv_nsec += 1000000000; // nsec/sec
    time_diff.tv_sec--;
  }

  // Convert the time to milliseconds
  *time = (float)(time_diff.tv_sec * 1000 + time_diff.tv_nsec / 1000000.0);
  return 0;
}

int time_and_fill_histogram_gpu(atoms_data *atoms, histogram *hist,
                                unsigned int block_size, float *time,
                                kernel_algorithm algorithm) {
  // Check if CUDA device is available
  int device_count;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    fprintf(stderr, "No CUDA devices available\n");
    return -1;
  }

  // Initialize data on the GPU
  atoms_data atoms_gpu = {
      .x_pos = NULL, .y_pos = NULL, .z_pos = NULL, .len = atoms->len};
  histogram hist_gpu = {
      .arr = NULL, .len = hist->len, .resolution = hist->resolution};
  CHECK_CUDA_ERROR(cudaMalloc(&atoms_gpu.x_pos, sizeof(double) * atoms->len));
  CHECK_CUDA_ERROR(cudaMalloc(&atoms_gpu.y_pos, sizeof(double) * atoms->len));
  CHECK_CUDA_ERROR(cudaMalloc(&atoms_gpu.z_pos, sizeof(double) * atoms->len));
  CHECK_CUDA_ERROR(cudaMalloc(&hist_gpu.arr, sizeof(bucket) * hist->len));

  // Copy data to GPU
  CHECK_CUDA_ERROR(cudaMemcpy(atoms_gpu.x_pos, atoms->x_pos,
                              sizeof(double) * atoms->len,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(atoms_gpu.y_pos, atoms->y_pos,
                              sizeof(double) * atoms->len,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(atoms_gpu.z_pos, atoms->z_pos,
                              sizeof(double) * atoms->len,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(hist_gpu.arr, hist->arr,
                              sizeof(bucket) * hist->len,
                              cudaMemcpyHostToDevice));
  int success;
  switch (algorithm) {
  case GRID_2D: {
    success = PDH_grid_2d(&atoms_gpu, &hist_gpu, block_size, time);
    break;
  }
  case SHARED_MEMORY: {
    success = PDH_shared_memory(&atoms_gpu, &hist_gpu, block_size, time);
    break;
  }
  case OUTPUT_PRIVATIZATION: {
    success = PDH_output_privatization(&atoms_gpu, &hist_gpu, block_size, time);
    break;
  }
  default: {
    fprintf(stderr, "Unknown algorithm\n");
    return -1;
  }
  }

  if (success != 0) {
    fprintf(stderr, "Error running the algorithm on the GPU\n");
    CHECK_CUDA_ERROR(cudaFree(atoms_gpu.y_pos));
    CHECK_CUDA_ERROR(cudaFree(atoms_gpu.x_pos));
    CHECK_CUDA_ERROR(cudaFree(atoms_gpu.z_pos));
    CHECK_CUDA_ERROR(cudaFree(hist_gpu.arr));

    return -1;
  }

  // Copy the histogram back to the CPU
  CHECK_CUDA_ERROR(cudaMemcpy(hist->arr, hist_gpu.arr,
                              sizeof(bucket) * hist->len,
                              cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(atoms_gpu.y_pos));
  CHECK_CUDA_ERROR(cudaFree(atoms_gpu.x_pos));
  CHECK_CUDA_ERROR(cudaFree(atoms_gpu.z_pos));
  CHECK_CUDA_ERROR(cudaFree(hist_gpu.arr));

  return 0;
}

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
