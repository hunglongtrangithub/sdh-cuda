#include <stdarg.h>
#include <stdio.h>

int sum(int head, int count, ...) {
  printf("Head: %d\n", head);
  va_list args;
  va_start(args, count);

  int result = 0;
  for (int i = 0; i < count; i++) {
    result += va_arg(args, int);
  }

  va_end(args);
  return result;
}

int main() {
  int result1 = sum(3, 1, 2, 3);
  int result2 = sum(5, 10, 20, 30, 40, 50);
  int result3 = sum(3, 1);

  printf("Sum 1: %d\n", result1); // Output: Sum 1: 6
  printf("Sum 2: %d\n", result2); // Output: Sum 2: 150

  return 0;
}
