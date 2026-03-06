# Spatial Distance Histogram (SDH) on CUDA GPUs

> **Note:** This is a Zig migration from the original C code, which has better numerical safety. For the original C implementation, check out the `c` branch.

## Overview

This project implements and benchmarks algorithms for computing 2-body statistics on GPUs, as described in the paper [**"Algorithms and Framework for Computing 2-body Statistics on GPUs"**](https://cse.usf.edu/~tuy/pub/DAPD19.pdf). The primary goal is to compare the execution speed of various GPU implementations against a CPU version, analyzing performance under different CUDA block sizes.

## Implemented Methods

The following algorithms are implemented:

- **CPU Baseline:** A straightforward nested loop implementation for computing the spatial distance histogram (SDH) in a serial fashion.
- **GPU 2D Grid Version:** Uses a 2D grid of threads to compute the SDH.
- **GPU Shared Memory Version:** Optimized to use shared memory for better memory access efficiency.
- **GPU Shared Memory with Output Privatization + Reduction:** Further optimized by reducing contention through privatization and reduction techniques.

## Compilation

To compile the project, use Zig (version 0.15.2):

```sh
zig build
```

This will generate the executable in the `zig-out/bin/` directory as `sdh`.

## Running the Program

### For Experiment

```sh
./zig-out/bin/sdh_cuda experiment [csv_path]
```

The optional `csv_path` defaults to `experiment_results.csv`.

### Generate Plots

After running the experiment, generate plots from the results. First, install dependencies:

```sh
uv sync
```

Then run the plotting script:

```sh
uv run plot.py [csv_path]
```

The optional `csv_path` defaults to `experiment_results.csv`. Plots are saved to the `plots/` directory.

### For Demo

The executable requires a subcommand and three command-line arguments:

```sh
./zig-out/bin/sdh demo <num_particles> <bucket_width> <block_size> [--gpu-only]
```

Example:

```sh
./zig-out/bin/sdh demo 10000 500 32
```

- `num_particles` → Number of particles to compute SDH for.
- `bucket_width` → Bin width for the histogram.
- `block_size` → CUDA block size used in the GPU kernel.
- `--gpu-only` → (Optional) Run only GPU implementations, skipping CPU.

## Expected Output

The program prints:

- The computed SDH for each method.
- Execution times for the CPU and GPU implementations.
- Speedup comparisons between CPU and GPU.

Check out the `plots/` directory to see the execution time of all GPU implementations as a function of block size, number of atoms, and resolution.

## References

- Napath Pitaksirianan, Zhila Nouri, and Yi-Cheng Tu. "Efficient 2-Body Statistics Computation on GPUs: Parallelization & Beyond." _Proceedings of the 45th International Conference on Parallel Processing_, pp. 380-385, 2016.
