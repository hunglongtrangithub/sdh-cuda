#include "histogram.h"
#include <stdio.h>

void histogram_init(histogram *hist) {
  // Set all counts to 0
  for (size_t i = 0; i < hist->len; i++) {
    hist->arr[i].d_cnt = 0;
  }
}

void display_histogram(histogram *hist) {
  long long unsigned int total_cnt = 0;

  for (size_t i = 0; i < hist->len; i++) {
    if (i % 5 == 0)
      printf("\n%02zu: ", i);

    printf("%15lld ", hist->arr[i].d_cnt);
    total_cnt += hist->arr[i].d_cnt;

    if (i == hist->len - 1)
      printf("\n T:%lld \n", total_cnt);
    else
      printf("| ");
  }
}
