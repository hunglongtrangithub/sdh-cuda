#include <stdio.h>
#include <stdlib.h>

int main() {
  size_t length = 10;
  double *arr = (double *)malloc(3 * length * sizeof(double));
  if (arr == NULL) {
    return 1; // Memory allocation failed
  }
  double *arr1 = arr;
  double *arr2 = arr + length;
  double *arr3 = arr + 2 * length;

  for (size_t i = 0; i < length; i++) {
    arr1[i] = i;
    arr2[i] = i + 1;
    arr3[i] = i + 2;
  }

  // Print the arrays
  for (size_t i = 0; i < length; i++) {
    printf("arr1[%zu] = %f, arr2[%zu] = %f, arr3[%zu] = %f\n", i, arr1[i], i,
           arr2[i], i, arr3[i]);
  }

  free(arr);
  return 0;
}
