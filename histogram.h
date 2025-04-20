#ifndef HISTOGRAM
#define HISTOGRAM

#include <stdint.h>

typedef struct hist_entry {
  unsigned long long d_cnt;
} bucket;

typedef struct histogram {
  bucket *arr;
  uint64_t len;
  double resolution;
} histogram;

void histogram_init(histogram *hist);
void display_histogram(histogram *hist);
void histogram_cleanup(histogram *hist);

#endif // !HISTOGRAM
