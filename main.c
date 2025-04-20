#include "experiment.h"
#include "utils.h"
#include <errno.h>
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

  histogram hist_cpu = {buckets_cpu, num_buckets, resolution};
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

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <num_particles> <bucket_width> <block_size>\n"
          "  %s experiment\n",
          prog, prog);
}

/* ---------- helpers ---------------------------------------------------- */
static int parse_u64(const char *s, uint64_t *out) {
  errno = 0;
  char *end;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno == ERANGE || *end || v == 0)
    return -1;
  *out = (uint64_t)v;
  return 0;
}
static int parse_double_pos(const char *s, double *out) {
  errno = 0;
  char *end;
  double v = strtod(s, &end);
  if (errno == ERANGE || *end || v <= 0.0)
    return -1;
  *out = v;
  return 0;
}

/* ---------- main ------------------------------------------------------- */
int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "experiment") == 0) {
    return experiment();
  }
  if (argc != 4) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  uint64_t particles, block;
  double bucket;

  if (parse_u64(argv[1], &particles) != 0) {
    fprintf(stderr, "Invalid particle count\n");
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (parse_double_pos(argv[2], &bucket) != 0) {
    fprintf(stderr, "Invalid bucket width\n");
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (parse_u64(argv[3], &block) != 0) {
    fprintf(stderr, "Invalid block size\n");
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  return demo(particles, bucket, block);
}
