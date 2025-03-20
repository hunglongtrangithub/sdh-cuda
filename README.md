# Spatial Distance Histogram (SDH) on CUDA GPUs

## Overview

This project implements and benchmarks algorithms for computing 2-body statistics on GPUs, as described in the paper [**"Algorithms and Framework for Computing 2-body Statistics on GPUs"**](https://cse.usf.edu/~tuy/pub/DAPD19.pdf). The primary goal is to compare the execution speed of various GPU implementations against a CPU version, analyzing performance under different CUDA block sizes.

## Implemented Methods

The following algorithms are implemented:

- **CPU Baseline:** A straightforward nested loop implementation for computing the spatial distance histogram (SDH) in a serial fashion.
- **GPU 2D Grid Version:** Uses a 2D grid of threads to compute the SDH.
- **GPU Shared Memory Version:** Optimized to use shared memory for better memory access efficiency.
- **GPU Shared Memory with Output Privatization + Reduction:** Further optimized by reducing contention through privatization and reduction techniques.

## Compilation

To compile the project, use the provided `Makefile`. Ensure you have **CUDA** installed before proceeding.

```sh
make -j$(nproc)
```

This will generate the executable in the `bin/` directory as `run`.

## Running the Program

The executable requires three command-line arguments:

```sh
./bin/run {#of_samples} {bucket_width} {block_size}
```

Example:

```sh
./bin/run 10000 500 32
```

- `#of_samples` → Number of particles to compute SDH for.
- `bucket_width` → Bin width for the histogram.
- `block_size` → CUDA block size used in the GPU kernel.

## Expected Output

The program prints:

- The computed SDH for each method.
- Execution times for the CPU and GPU implementations.
- Speedup comparisons between CPU and GPU.

## TODO

- [ ] **Plot Execution Time vs Block Size:** Generate a graph to visualize the execution time of all GPU implementations as a function of block size.

## References

- Napath Pitaksirianan, Zhila Nouri, and Yi-Cheng Tu. "Efficient 2-Body Statistics Computation on GPUs: Parallelization & Beyond." _Proceedings of the 45th International Conference on Parallel Processing_, pp. 380-385, 2016.
