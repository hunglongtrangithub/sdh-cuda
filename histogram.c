#include "histogram.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void histogram_init(histogram *hist) {
  // Set all counts to 0
  for (uint64_t i = 0; i < hist->len; i++) {
    hist->arr[i].d_cnt = 0;
  }
}

void display_histogram(histogram *hist) {
  uint64_t total_cnt = 0;

  for (uint64_t i = 0; i < hist->len; i++) {
    if (i % 5 == 0)
      printf("\n%02llu: ", i);

    printf("%15lld ", hist->arr[i].d_cnt);
    total_cnt += hist->arr[i].d_cnt;

    if (i == hist->len - 1)
      printf("\n T:%llu \n", total_cnt);
    else
      printf("| ");
  }
}

void histogram_cleanup(histogram *hist) {
  free(hist->arr);
  hist->arr = NULL;
  hist->len = 0;
  hist->resolution = 0.0;
}
