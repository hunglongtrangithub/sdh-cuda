#define CHECK_CUDA_ERROR(call)                                                 \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      return -1;                                                               \
    }                                                                          \
  }
