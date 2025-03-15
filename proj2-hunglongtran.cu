#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BOX_SIZE 23000
/* CUDA error checking helper function */
#define CHECK_CUDA_ERROR(call)                                                 \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      return -1;                                                               \
    }                                                                          \
  }

enum platform { CPU, GPU };

enum kernel_algorithm { BASELINE, OPTIMIZED };

typedef struct atomdesc {
  double x_pos;
  double y_pos;
  double z_pos;
} atom;

typedef struct hist_entry {
  unsigned long long d_cnt;
} bucket;

typedef struct histogram {
  bucket *arr;
  unsigned int len;
  double resolution;
} histogram;

typedef struct atom_list {
  double *x_pos;
  double *y_pos;
  double *z_pos;
  unsigned long long len;
} atoms_data;

/* Core SDH algorithm. Mutates the histogram */
int PDH_baseline(atoms_data *atoms, histogram *hist) {
  int i, j;
  double dist;

  for (i = 0; i < atoms->len; i++) {
    for (j = i + 1; j < atoms->len; j++) {
      dist = sqrt((atoms->x_pos[i] - atoms->x_pos[j]) *
                      (atoms->x_pos[i] - atoms->x_pos[j]) +
                  (atoms->y_pos[i] - atoms->y_pos[j]) *
                      (atoms->y_pos[i] - atoms->y_pos[j]) +
                  (atoms->z_pos[i] - atoms->z_pos[j]) *
                      (atoms->z_pos[i] - atoms->z_pos[j]));
      int h_pos = (int)(dist / hist->resolution);
      if (h_pos >= hist->len)
        continue;
      hist->arr[h_pos].d_cnt++;
    }
  }

  return 0;
}

/* CUDA PDH kernel */
__global__ void PDH_cuda_kernel(double *x_pos, double *y_pos, double *z_pos,
                                unsigned long long atoms_len, bucket *hist,
                                unsigned int hist_len, double resolution) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  // Filter out-of-bound threads and threads that have x >= y
  if (x >= atoms_len || y >= atoms_len || x >= y)
    return;

  // Calculate the distance between the two atoms
  double dist = sqrt((x_pos[x] - x_pos[y]) * (x_pos[x] - x_pos[y]) +
                     (y_pos[x] - y_pos[y]) * (y_pos[x] - y_pos[y]) +
                     (z_pos[x] - z_pos[y]) * (z_pos[x] - z_pos[y]));

  // Calculate the histogram position
  int h_pos = (int)(dist / resolution);
  if (h_pos >= hist_len)
    return;

  // Increment the histogram count with atomic operation
  atomicAdd(&hist[h_pos].d_cnt, 1);
}

__global__ void PDH_cuda_kernel_optimized(double *x_pos, double *y_pos,
                                          double *z_pos,
                                          unsigned long long atoms_len,
                                          bucket *hist, unsigned int hist_len,
                                          double resolution) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= atoms_len)
    return;
  extern __shared__ double x_shared[];
  extern __shared__ double y_shared[];
  extern __shared__ double z_shared[];

  double x = x_pos[idx];
  double y = y_pos[idx];
  double z = z_pos[idx];

  // Loop through each next block from the current block
  for (int i = blockIdx.x + 1; i < gridDim.x; i++) {
    // Load the block into shared memory
    if (blockDim.x * i + threadIdx.x < atoms_len) {
      x_shared[threadIdx.x] = x_pos[blockDim.x * i + threadIdx.x];
      y_shared[threadIdx.x] = y_pos[blockDim.x * i + threadIdx.x];
      z_shared[threadIdx.x] = z_pos[blockDim.x * i + threadIdx.x];
    }
    __syncthreads();

    // Loop through each atom in the cached block
    for (int j = 0; j < blockDim.x; j++) {
      // Stop if the thread is out of bounds
      if (blockDim.x * i + j >= atoms_len)
        break;
      double dist = sqrt((x - x_shared[j]) * (x - x_shared[j]) +
                         (y - y_shared[j]) * (y - y_shared[j]) +
                         (z - z_shared[j]) * (z - z_shared[j]));
      int h_pos = (int)(dist / resolution);
      if (h_pos < hist_len) {
        atomicAdd(&hist[h_pos].d_cnt, 1);
      }
    }
  }
}

/* CUDA PDH algorithm. Mutates the histogram */
int PDH_cuda(atoms_data *atoms_gpu, histogram *hist_gpu, int block_size,
             float *diff, kernel_algorithm algorithm) {
  // Check if CUDA device is available
  int deviceCount;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "No CUDA devices found\n");
    return -1;
  }

  switch (algorithm) {
  case BASELINE: {
    printf("Running baseline kernel\n");
    // Define the number of blocks and threads per block
    // Maximum x, y dimensions of a block are typically 1024 threads
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((atoms_gpu->len + block_dim.x - 1) / block_dim.x,
                  (atoms_gpu->len + block_dim.y - 1) / block_dim.y);
    printf("Grid dimensions: %d x %d\n", grid_dim.x, grid_dim.y);
    printf("Block dimensions: %d x %d\n", block_dim.x, block_dim.y);

    cudaEvent_t start_time, end_time;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_time));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_time));
    CHECK_CUDA_ERROR(cudaEventRecord(start_time, 0));

    // Launch the kernel
    PDH_cuda_kernel<<<grid_dim, block_dim>>>(
        atoms_gpu->x_pos, atoms_gpu->y_pos, atoms_gpu->z_pos, atoms_gpu->len,
        hist_gpu->arr, hist_gpu->len, hist_gpu->resolution);
    CHECK_CUDA_ERROR(cudaEventRecord(end_time, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_time));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(diff, start_time, end_time));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_time));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_time));
  }
  case OPTIMIZED: {
    printf("Running optimized kernel\n");
    int grid_size = (atoms_gpu->len + block_size - 1) / block_size;
    int shared_mem_size = block_size * sizeof(double);
    printf("Grid size: %d\n", grid_size);
    printf("Block size: %d\n", block_size);
    printf("Shared memory size: %d\n", shared_mem_size);

    cudaEvent_t start_time, end_time;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_time));
    CHECK_CUDA_ERROR(cudaEventCreate(&end_time));
    CHECK_CUDA_ERROR(cudaEventRecord(start_time, 0));

    // Launch the kernel
    PDH_cuda_kernel_optimized<<<grid_size, block_size, shared_mem_size>>>(
        atoms_gpu->x_pos, atoms_gpu->y_pos, atoms_gpu->z_pos, atoms_gpu->len,
        hist_gpu->arr, hist_gpu->len, hist_gpu->resolution);
    CHECK_CUDA_ERROR(cudaEventRecord(end_time, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end_time));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(diff, start_time, end_time));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_time));
    CHECK_CUDA_ERROR(cudaEventDestroy(end_time));
  }
  }

  return 0;
}

/* Histogram output function */
void display_histogram(histogram *hist) {
  long long total_cnt = 0;

  for (int i = 0; i < hist->len; i++) {
    if (i % 5 == 0)
      printf("\n%02d: ", i);

    printf("%15lld ", hist->arr[i].d_cnt);
    total_cnt += hist->arr[i].d_cnt;

    if (i == hist->len - 1)
      printf("\n T:%lld \n", total_cnt);
    else
      printf("| ");
  }
}

struct timespec calculate_time(const struct timespec *start,
                               const struct timespec *end) {
  struct timespec diff = {.tv_sec = end->tv_sec - start->tv_sec, //
                          .tv_nsec = end->tv_nsec - start->tv_nsec};
  if (diff.tv_nsec < 0) {
    diff.tv_nsec += 1000000000; // nsec/sec
    diff.tv_sec--;
  }
  return diff;
}

/* Timing and histogram filling function. The algorithm mutates the histogram
 */
int time_and_fill_histogram_cpu(atoms_data *atoms, histogram *hist,
                                int (*algorithm)(atoms_data *, histogram *),
                                struct timespec *diff) {
  struct timespec start_time;
  struct timespec end_time;

  // Record the start time
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  if (algorithm(atoms, hist) != 0) {
    return -1;
  }

  // Record the end time
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  // Calculate the time difference
  *diff = calculate_time(&start_time, &end_time);
  return 0;
}

int time_and_fill_histogram_gpu(atoms_data *atoms, histogram *hist,
                                int block_size, float *diff,
                                kernel_algorithm algorithm,
                                int (*kernel)(atoms_data *, histogram *,
                                              int block_size, float *diff,
                                              kernel_algorithm algorithm)) {
  if (kernel(atoms, hist, block_size, diff, algorithm) != 0) {
    return -1;
  }
  return 0;
}

/* Results calculation and display function */
int calculate_and_display_histogram(atoms_data *atoms, histogram *hist,
                                    platform platform, float *time, int count,
                                    ...) {
  switch (platform) {
  case CPU: {
    printf("Running CPU version\n");
    struct timespec time_diff;
    // Do the calculation and get the time
    if (time_and_fill_histogram_cpu(atoms, hist, PDH_baseline, &time_diff) !=
        0) {
      fprintf(stderr, "Error running the algorithm on the CPU\n");
      return -1;
    }

    *time = (double)(time_diff.tv_sec * 1000 + time_diff.tv_nsec / 1000000.0);
    return 0;
  }
  case GPU: {
    printf("Running GPU version\n");

    va_list args;
    va_start(args, count);
    int block_size = va_arg(args, int);
    int algorithm_int = va_arg(args, int);
    kernel_algorithm algorithm = static_cast<kernel_algorithm>(algorithm_int);
    va_end(args);

    // Check if CUDA device is available
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
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

    // Copy data to GPU with error checking
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

    // Do the calculation and get the time
    if (time_and_fill_histogram_gpu(&atoms_gpu, &hist_gpu, block_size, time,
                                    algorithm, PDH_cuda) != 0) {
      fprintf(stderr, "Error running the algorithm on the GPU\n");
      CHECK_CUDA_ERROR(cudaFree(atoms_gpu.x_pos));
      CHECK_CUDA_ERROR(cudaFree(atoms_gpu.y_pos));
      CHECK_CUDA_ERROR(cudaFree(atoms_gpu.z_pos));
      CHECK_CUDA_ERROR(cudaFree(hist_gpu.arr));

      return -1;
    }

    // Copy the histogram back to the CPU
    CHECK_CUDA_ERROR(cudaMemcpy(hist->arr, hist_gpu.arr,
                                sizeof(bucket) * hist->len,
                                cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(atoms_gpu.x_pos));
    CHECK_CUDA_ERROR(cudaFree(atoms_gpu.y_pos));
    CHECK_CUDA_ERROR(cudaFree(atoms_gpu.z_pos));
    CHECK_CUDA_ERROR(cudaFree(hist_gpu.arr));

    return 0;
  }
  }

  // Should never reach here
  return -1;
}

/* Atoms data generation function */
atoms_data atoms_data_init(unsigned int count, int box_size) {
  atoms_data atoms = {
      .x_pos = (double *)malloc(count * sizeof(double)),
      .y_pos = (double *)malloc(count * sizeof(double)),
      .z_pos = (double *)malloc(count * sizeof(double)),
      .len = count,
  };

  // Generate random data points
  srand(1); // Fixed seed for reproducibility
  for (int i = 0; i < atoms.len; i++) {
    atoms.x_pos[i] = ((double)(rand()) / RAND_MAX) * box_size;
    atoms.y_pos[i] = ((double)(rand()) / RAND_MAX) * box_size;
    atoms.z_pos[i] = ((double)(rand()) / RAND_MAX) * box_size;
  }

  return atoms;
}

/* Histogram initialization function */
histogram histogram_init(double resolution, int box_size) {
  // The maximum distance between two points in a box is the diagonal
  unsigned int num_buckets =
      (unsigned int)(box_size * sqrt(3) / resolution) + 1;
  // Allocate the histogram array to store the counts. Initialize to zero
  bucket *hist_arr = (bucket *)calloc(num_buckets, sizeof(bucket));
  if (hist_arr == NULL) {
    fprintf(stderr, "Error allocating memory for histogram\n");
    exit(1);
  }
  histogram hist = {
      hist_arr,
      num_buckets,
      resolution,
  };

  return hist;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s {#of_samples} {bucket_width} {block_size}\n", argv[0]);
    return 1;
  }

  unsigned int particle_count = atoi(argv[1]);
  double resolution = atof(argv[2]);
  int block_size = atoi(argv[3]);

  // Generate heap-allocated data
  atoms_data atoms = atoms_data_init(particle_count, BOX_SIZE);
  histogram hist = histogram_init(resolution, BOX_SIZE);

  // Run algorithms
  float time_gpu_baseline, time_gpu_optimized;
  if (calculate_and_display_histogram(&atoms, &hist, GPU, &time_gpu_baseline, 1,
                                      block_size, BASELINE) != 0) {
    printf("Error running GPU baseline version. Exiting\n");
    free(hist.arr);
    free(atoms.x_pos);
    free(atoms.y_pos);
    free(atoms.z_pos);
    return 1;
  }
  if (calculate_and_display_histogram(&atoms, &hist, GPU, &time_gpu_optimized,
                                      1, block_size, OPTIMIZED) != 0) {
    printf("Error running GPU optimized version. Exiting\n");
    free(hist.arr);
    free(atoms.x_pos);
    free(atoms.y_pos);
    free(atoms.z_pos);
    return 1;
  }

  // Display timing results
  printf("GPU time in miliseconds (baseline): %f\n", time_gpu_baseline);
  printf("GPU time in miliseconds (optimized): %f\n", time_gpu_optimized);

  free(hist.arr);
  free(atoms.x_pos);
  free(atoms.y_pos);
  free(atoms.z_pos);

  return 0;
}
