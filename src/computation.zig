const std = @import("std");
const log = std.log.scoped(.computation);
const kernels = @import("kernels.zig");

const Atom = @import("Atom.zig");
const Histogram = @import("Histogram.zig");
const Bucket = Histogram.Bucket;

const c = @cImport({
    @cInclude("cuda_runtime.h");
});

// CPU

pub fn timeAndFillHistogramCpu(atoms: *const Atom, hist: *Histogram) std.time.Timer.Error!u64 {
    var timer = try std.time.Timer.start();
    const len = atoms.len;
    const res = hist.resolution;
    const num_buckets = hist.arr.len;

    for (0..len) |i| {
        const x1 = atoms.x_pos[i];
        const y1 = atoms.y_pos[i];
        const z1 = atoms.z_pos[i];

        for (i + 1..len) |j| {
            const dx = x1 - atoms.x_pos[j];
            const dy = y1 - atoms.y_pos[j];
            const dz = z1 - atoms.z_pos[j];

            const dist = @sqrt(dx * dx + dy * dy + dz * dz);
            const h_pos: usize = @intFromFloat(dist / res);

            if (h_pos < num_buckets) {
                hist.arr[h_pos].d_cnt += 1;
            }
        }
    }
    return timer.read();
}

// GPU helpers

fn cudaMalloc(comptime T: type, len: usize) CudaError![]T {
    var ptr: ?*anyopaque = null;
    try check(c.cudaMalloc(&ptr, len * @sizeOf(T)));
    const casted_ptr: [*]T = @ptrCast(@alignCast(ptr.?));
    return casted_ptr[0..len];
}

fn cudaFree(comptime T: type, ptr: [*]T) void {
    _ = c.cudaFree(@ptrCast(ptr));
}

fn cudaMemcpy(
    comptime T: type,
    dst: [*]T,
    src: []const T,
    kind: c.enum_cudaMemcpyKind,
) CudaError!void {
    try check(c.cudaMemcpy(
        @ptrCast(dst),
        @ptrCast(src.ptr),
        src.len * @sizeOf(T),
        kind,
    ));
}

// GPU

pub const KernelAlgorithm = enum {
    grid_2d,
    shared_memory,
    output_privatization,
};

const CudaError = error{
    /// A CUDA runtime API call returned a non-success status code.
    /// The specific CUDA error string is logged before this error is returned.
    /// This is a catch-all for any underlying CUDA failure (e.g. invalid arguments,
    /// driver errors, failed kernel launches, synchronization failures, or
    /// memory allocation/copy errors).
    /// Action: check the logged CUDA error message for details.
    CudaFailed,

    /// `cudaGetDeviceCount` reported zero CUDA-capable devices.
    /// This means no compatible GPU was detected by the CUDA runtime.
    /// Action: ensure a CUDA-capable GPU is installed and that the correct
    /// drivers are loaded.
    NoCudaDevices,

    /// The requested block size exceeds the device's `maxThreadsPerBlock` limit,
    /// or an intermediate value derived from the block size (e.g. the 2D block
    /// side length, shared memory calculation, or reduction block size) overflowed
    /// or could not be represented as a u32.
    /// Action: reduce the `block_size` argument. Typical maximums are 512 or 1024
    /// threads per block depending on the GPU architecture.
    BlockSizeTooLarge,

    /// The computed grid dimension exceeds the device's `maxGridSize` limit, or
    /// the grid dimension could not be represented as a u32.
    /// This can happen when the number of atoms is very large relative to the
    /// block size, producing a grid that is too wide for the device.
    /// Action: increase the block size or reduce the number of atoms.
    GridSizeTooLarge,

    /// The amount of dynamic shared memory required by the kernel exceeds the
    /// device's `sharedMemPerBlock` limit. Shared memory is used to cache atom
    /// positions (shared_memory kernel) or atom positions plus a private histogram
    /// copy (output_privatization kernel).
    /// Action: reduce the block size, increase the histogram resolution (fewer
    /// buckets), or use a different algorithm that requires less shared memory.
    SharedMemoryTooLarge,

    /// The histogram length (number of buckets) is too large to be represented
    /// as a u32 grid dimension for the reduction kernel, or it caused an overflow
    /// when computing shared memory or the 2D histogram size.
    /// Action: increase the histogram resolution value (bucket width) so that
    /// fewer buckets are produced.
    HistLenTooLarge,

    /// The product of `hist_len * grid_size` overflowed when computing the size
    /// of the 2D privatized histogram array, meaning the required device memory
    /// allocation would be impossibly large.
    /// Action: increase the histogram resolution (fewer buckets), increase the
    /// block size (fewer grid blocks), or reduce the number of atoms.
    OutOfDeviceMemory,
};

fn check(result: c.cudaError_t) CudaError!void {
    if (result != c.cudaSuccess) {
        log.err("CUDA error: {s}", .{c.cudaGetErrorString(result)});
        return CudaError.CudaFailed;
    }
}

pub fn timeAndFillHistogramGpu(
    atoms: *const Atom,
    hist: *Histogram,
    block_size: usize,
    algorithm: KernelAlgorithm,
) CudaError!f32 {
    // Check for CUDA devices
    var device_count: c_int = 0;
    try check(c.cudaGetDeviceCount(&device_count));
    if (device_count == 0) return CudaError.NoCudaDevices;

    var prop: c.cudaDeviceProp = undefined;
    try check(c.cudaGetDeviceProperties(&prop, 0));

    if (std.math.cast(usize, prop.maxThreadsPerBlock)) |val| {
        if (block_size > val) return CudaError.BlockSizeTooLarge;
    }

    const atoms_len = atoms.len;
    const hist_len = hist.arr.len;
    const resolution = hist.resolution;

    // Allocate GPU memory for atoms
    const d_x = try cudaMalloc(f64, atoms.len);
    defer cudaFree(f64, d_x.ptr);
    const d_y = try cudaMalloc(f64, atoms.len);
    defer cudaFree(f64, d_y.ptr);
    const d_z = try cudaMalloc(f64, atoms.len);
    defer cudaFree(f64, d_z.ptr);

    // Copy atoms to device
    try cudaMemcpy(f64, d_x.ptr, atoms.x_pos[0..atoms.len], c.cudaMemcpyHostToDevice);
    try cudaMemcpy(f64, d_y.ptr, atoms.y_pos[0..atoms.len], c.cudaMemcpyHostToDevice);
    try cudaMemcpy(f64, d_z.ptr, atoms.z_pos[0..atoms.len], c.cudaMemcpyHostToDevice);

    // Allocate GPU memory for histogram
    const d_hist = try cudaMalloc(Bucket, hist.arr.len);
    defer cudaFree(Bucket, d_hist.ptr);
    try cudaMemcpy(Bucket, d_hist.ptr, hist.arr, c.cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    var start_event: c.cudaEvent_t = null;
    var end_event: c.cudaEvent_t = null;
    try check(c.cudaEventCreate(&start_event));
    defer _ = c.cudaEventDestroy(start_event);
    try check(c.cudaEventCreate(&end_event));
    defer _ = c.cudaEventDestroy(end_event);

    try check(c.cudaEventRecord(start_event, null));

    // Dispatch kernel
    switch (algorithm) {
        .grid_2d => {
            const side = std.math.sqrt(block_size);

            const grid_dim = std.math.divCeil(
                usize,
                atoms_len,
                side,
            ) catch 0;

            const side_u32 = std.math.cast(u32, side) orelse return CudaError.BlockSizeTooLarge;
            const grid_u32 = std.math.cast(u32, grid_dim) orelse return CudaError.GridSizeTooLarge;

            log.info("Grid 2D: grid=({d},{d}), block=({d},{d})", .{ grid_u32, grid_u32, side_u32, side_u32 });

            kernels.launch_grid_2d(
                d_x.ptr,
                d_y.ptr,
                d_z.ptr,
                atoms_len,
                d_hist.ptr,
                hist_len,
                resolution,
                grid_u32,
                grid_u32,
                1,
                side_u32,
                side_u32,
                1,
            );
            try check(c.cudaPeekAtLastError());
            try check(c.cudaDeviceSynchronize());
        },
        .shared_memory => {
            const grid_size = std.math.divCeil(
                usize,
                atoms_len,
                block_size,
            ) catch 0;
            const shared_mem_size = std.math.mul(
                usize,
                3 * @sizeOf(f64),
                block_size,
            ) catch return CudaError.BlockSizeTooLarge;

            if (shared_mem_size > prop.sharedMemPerBlock) return CudaError.SharedMemoryTooLarge;

            const grid_u32 = std.math.cast(u32, grid_size) orelse return CudaError.GridSizeTooLarge;
            const block_u32 = std.math.cast(u32, block_size) orelse return CudaError.BlockSizeTooLarge;

            log.info("Shared memory: grid={d}, block={d}, shmem={d}", .{ grid_u32, block_u32, shared_mem_size });

            kernels.launch_shared_memory(
                d_x.ptr,
                d_y.ptr,
                d_z.ptr,
                atoms_len,
                d_hist.ptr,
                hist_len,
                resolution,
                grid_u32,
                block_u32,
                shared_mem_size,
            );
            try check(c.cudaPeekAtLastError());
            try check(c.cudaDeviceSynchronize());
        },
        .output_privatization => {
            const grid_size = std.math.divCeil(
                usize,
                atoms_len,
                block_size,
            ) catch 0;

            if (std.math.cast(usize, prop.maxGridSize[0])) |max_grid| {
                if (grid_size > max_grid) return CudaError.GridSizeTooLarge;
            }

            const shared_mem_size = std.math.add(
                usize,
                std.math.mul(usize, 3 * @sizeOf(f64), block_size) catch return CudaError.BlockSizeTooLarge,
                std.math.mul(usize, @sizeOf(Bucket), hist_len) catch return CudaError.HistLenTooLarge,
            ) catch return CudaError.SharedMemoryTooLarge;
            if (shared_mem_size > prop.sharedMemPerBlock) return CudaError.SharedMemoryTooLarge;

            const hist_2d_len = std.math.mul(usize, hist_len, grid_size) catch return CudaError.OutOfDeviceMemory;
            const d_hist_2d = try cudaMalloc(Bucket, hist_2d_len);
            defer cudaFree(Bucket, d_hist_2d.ptr);

            const grid_u32 = std.math.cast(u32, grid_size) orelse return CudaError.GridSizeTooLarge;
            const block_u32 = std.math.cast(u32, block_size) orelse return CudaError.BlockSizeTooLarge;

            log.info("Output privatization: grid={d}, block={d}, shmem={d}", .{ grid_u32, block_u32, shared_mem_size });

            kernels.launch_output_privatization(
                d_x.ptr,
                d_y.ptr,
                d_z.ptr,
                atoms_len,
                d_hist_2d.ptr,
                hist_len,
                resolution,
                grid_u32,
                block_u32,
                shared_mem_size,
            );
            try check(c.cudaPeekAtLastError());
            try check(c.cudaDeviceSynchronize());

            // Reduction pass
            const red_block_usize = std.math.ceilPowerOfTwo(usize, block_size) catch return CudaError.BlockSizeTooLarge;
            const red_block = std.math.cast(u32, red_block_usize) orelse return CudaError.BlockSizeTooLarge;
            const red_shmem: usize = std.math.mul(usize, red_block_usize, @sizeOf(Bucket)) catch return CudaError.SharedMemoryTooLarge;
            if (red_shmem > prop.sharedMemPerBlock) return CudaError.SharedMemoryTooLarge;

            const red_grid = std.math.cast(u32, hist_len) orelse return CudaError.HistLenTooLarge;

            log.info("Reduction: grid={d}, block={d}, shmem={d}", .{ red_grid, red_block, red_shmem });

            kernels.launch_reduction(
                d_hist_2d.ptr,
                grid_size,
                d_hist.ptr,
                red_grid,
                red_block,
                red_shmem,
            );
            try check(c.cudaPeekAtLastError());
            try check(c.cudaDeviceSynchronize());
        },
    }

    // Stop timing
    try check(c.cudaEventRecord(end_event, null));
    try check(c.cudaEventSynchronize(end_event));

    var elapsed_ms: f32 = 0;
    try check(c.cudaEventElapsedTime(&elapsed_ms, start_event, end_event));

    // Copy histogram back to host
    try cudaMemcpy(Bucket, hist.arr.ptr, d_hist, c.cudaMemcpyDeviceToHost);

    return elapsed_ms;
}
