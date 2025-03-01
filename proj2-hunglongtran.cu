#include <cuda_runtime.h>
#include <driver_types.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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
  atom *arr;
  unsigned long long len;
} atoms_data;

/* Helper function to calculate distance between two points */
__host__ __device__ double p2p_distance(atom a1, atom a2) {
  double x1 = a1.x_pos;
  double x2 = a2.x_pos;
  double y1 = a1.y_pos;
  double y2 = a2.y_pos;
  double z1 = a1.z_pos;
  double z2 = a2.z_pos;

  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
              (z1 - z2) * (z1 - z2));
}

/* Core SDH algorithm. Mutates the histogram */
int PDH_baseline(atoms_data *atoms, histogram *hist) {
  int i, j;
  double dist;

  for (i = 0; i < atoms->len; i++) {
    for (j = i + 1; j < atoms->len; j++) {
      dist = p2p_distance(atoms->arr[i], atoms->arr[j]);
      int h_pos = (int)(dist / hist->resolution);
      if (h_pos >= hist->len)
        continue;
      hist->arr[h_pos].d_cnt++;
    }
  }

  return 0;
}

/* CUDA PDH kernel */
__global__ void PDH_cuda_kernel(atom *atoms, long long atoms_len, bucket *hist,
                                int hist_len, double resolution) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  // Filter out-of-bound threads and threads that have x >= y
  if (x >= atoms_len || y >= atoms_len || x >= y)
    return;

  // Calculate the distance between the two atoms
  double dist = p2p_distance(atoms[x], atoms[y]);

  // Calculate the histogram position
  int h_pos = (int)(dist / resolution);
  if (h_pos >= hist_len)
    return;

  // Increment the histogram count with atomic operation
  atomicAdd(&hist[h_pos].d_cnt, 1);
}

/* CUDA PDH algorithm. Mutates the histogram */
int PDH_cuda(atoms_data *atoms_gpu, histogram *hist_gpu, int block_size,
             float *diff) {
  // Check if CUDA device is available
  int deviceCount;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "No CUDA devices found\n");
    return -1;
  }

  // Define the number of blocks and threads per block
  // // Maximum x, y dimensions of a block are typically 1024 threads
  dim3 block_dim(block_size, block_size);
  dim3 grid_dim((atoms_gpu->len + block_dim.x - 1) / block_dim.x,
                (atoms_gpu->len + block_dim.y - 1) / block_dim.y);

  cudaEvent_t start_time, end_time;
  cudaEventCreate(&start_time);
  cudaEventCreate(&end_time);
  cudaEventRecord(start_time, 0);
  // Launch the kernel
  PDH_cuda_kernel<<<grid_dim, block_dim>>>(atoms_gpu->arr, atoms_gpu->len,
                                           hist_gpu->arr, hist_gpu->len,
                                           hist_gpu->resolution);
  cudaEventRecord(end_time, 0);
  cudaEventSynchronize(end_time);
  cudaEventElapsedTime(diff, start_time, end_time);
  cudaEventDestroy(start_time);
  cudaEventDestroy(end_time);

  // Check for kernel launch errors
  CHECK_CUDA_ERROR(cudaGetLastError());

  // Synchronize to ensure kernel completion
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

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
  struct timespec diff = {.tv_sec = start->tv_sec - end->tv_sec, //
                          .tv_nsec = start->tv_nsec - end->tv_nsec};
  if (diff.tv_nsec < 0) {
    diff.tv_nsec += 1000000000; // nsec/sec
    diff.tv_sec--;
  }
  return diff;
}

/* Timing and histogram filling function. The algorithm mutates the histogram */
int time_and_fill_histogram_cpu(atoms_data *atoms, histogram *hist,
                                int (*algorithm)(atoms_data *, histogram *),
                                struct timespec *diff) {
  struct timespec start_time;
  struct timespec end_time;
  if (algorithm(atoms, hist) != 0) {
    return -1;
  }

  *diff = calculate_time(&start_time, &end_time);
  return 0;
}

int time_and_fill_histogram_gpu(atoms_data *atoms, histogram *hist,
                                int block_size, float *diff,
                                int (*algorithm)(atoms_data *, histogram *,
                                                 int block_size, float *diff)) {
  if (algorithm(atoms, hist, block_size, diff) != 0) {
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

    // Display histogram
    display_histogram(hist);

    *time = (double)(time_diff.tv_sec * 1000 + time_diff.tv_nsec / 1000000.0);
    return 0;
  }
  case GPU: {
    printf("Running GPU version\n");

    va_list args;
    va_start(args, count);
    int block_size = va_arg(args, int);
    va_end(args);

    // Check if CUDA device is available
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
      fprintf(stderr, "No CUDA devices available\n");
      return -1;
    }

    // Initialize data on the GPU
    atom *atoms_arr_gpu;
    bucket *hist_arr_gpu;
    CHECK_CUDA_ERROR(cudaMalloc(&atoms_arr_gpu, sizeof(atom) * atoms->len));
    CHECK_CUDA_ERROR(cudaMalloc(&hist_arr_gpu, sizeof(bucket) * hist->len));

    // Copy data to GPU with error checking
    CHECK_CUDA_ERROR(cudaMemcpy(atoms_arr_gpu, atoms->arr,
                                sizeof(atom) * atoms->len,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(hist_arr_gpu, hist->arr,
                                sizeof(bucket) * hist->len,
                                cudaMemcpyHostToDevice));

    // Initialize the atoms and histogram data structures. Thsese will hold the
    // data on the GPU
    atoms_data atoms_gpu = {atoms_arr_gpu, atoms->len};
    histogram hist_gpu = {hist_arr_gpu, hist->len, hist->resolution};

    // Do the calculation and get the time
    if (time_and_fill_histogram_gpu(&atoms_gpu, &hist_gpu, block_size, time,
                                    PDH_cuda) != 0) {
      fprintf(stderr, "Error running the algorithm on the GPU\n");
      cudaFree(atoms_arr_gpu);
      cudaFree(hist_arr_gpu);
      return -1;
    }

    // Copy the histogram back to the CPU
    CHECK_CUDA_ERROR(cudaMemcpy(hist->arr, hist_gpu.arr,
                                sizeof(bucket) * hist->len,
                                cudaMemcpyDeviceToHost));

    cudaFree(atoms_arr_gpu);
    cudaFree(hist_arr_gpu);

    // Display histogram
    display_histogram(hist);

    return 0;
  }
  }

  // Should never reach here
  return -1;
}

/* Atoms data generation function */
atoms_data init_atoms_data(unsigned int count, int box_size) {
  atom *atoms_arr = (atom *)malloc(sizeof(atom) * count);
  if (atoms_arr == NULL) {
    fprintf(stderr, "Error allocating memory for atoms\n");
    exit(1);
  }
  atoms_data atoms = {
      atoms_arr,
      count,
  };

  // Generate random data points
  srand(1); // Fixed seed for reproducibility
  for (int i = 0; i < atoms.len; i++) {
    atoms.arr[i].x_pos = ((double)(rand()) / RAND_MAX) * box_size;
    atoms.arr[i].y_pos = ((double)(rand()) / RAND_MAX) * box_size;
    atoms.arr[i].z_pos = ((double)(rand()) / RAND_MAX) * box_size;
  }

  return atoms;
}

/* Histogram initialization function */
histogram init_histogram(double resolution, int box_size) {
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
  atoms_data atoms = init_atoms_data(particle_count, BOX_SIZE);
  histogram hist_cpu = init_histogram(resolution, BOX_SIZE);
  histogram hist_gpu = init_histogram(resolution, BOX_SIZE);

  // Run algorithms
  float time_cpu, time_gpu;
  if (calculate_and_display_histogram(&atoms, &hist_cpu, CPU, &time_cpu, 0) !=
      0) {
    printf("Error running CPU version. Exiting\n");
    free(hist_cpu.arr);
    free(atoms.arr);
    free(hist_gpu.arr);
    return 1;
  }
  if (calculate_and_display_histogram(&atoms, &hist_gpu, GPU, &time_gpu, 1,
                                      block_size) != 0) {
    printf("Error running GPU version. Exiting\n");
    free(hist_cpu.arr);
    free(atoms.arr);
    free(hist_gpu.arr);
    return 1;
  }

  // Calculate the diff histogram (stored in hist_cpu)
  for (int i = 0; i < hist_cpu.len; i++) {
    hist_cpu.arr[i].d_cnt -= hist_gpu.arr[i].d_cnt;
  }

  // Display timing results
  printf("CPU time in miliseconds: %f\n", time_cpu);
  printf("GPU time in miliseconds: %f\n", time_gpu);
  printf("Speedup: %f\n", time_cpu / time_gpu);

  // Display the diff histogram
  printf("Diff histogram:\n");
  display_histogram(&hist_cpu);

  free(hist_gpu.arr);
  free(atoms.arr);
  free(hist_cpu.arr);

  return 0;
}
