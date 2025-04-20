#include <cstdio>
#include <cstdlib>

void *check_malloc(size_t size) {
  void *ptr = malloc(size);
  if (ptr == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  return ptr;
}