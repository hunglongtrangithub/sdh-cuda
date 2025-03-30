#include "cuda_utils.h"
#include "kernels.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

// Function to calculate pairwise distance histogram for CPU
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

// Function to measure CPU time for PDH
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

// Function to measure GPU time for PDH
int time_and_fill_histogram_gpu(atoms_data *atoms, histogram *hist,
                                unsigned long int block_size, float *time,
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
