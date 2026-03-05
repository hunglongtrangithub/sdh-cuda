/// CUDA extern bridges
const Histogram = @import("Histogram.zig");
const Bucket = Histogram.Bucket;

pub extern fn launch_grid_2d(
    x_pos: [*]f64,
    y_pos: [*]f64,
    z_pos: [*]f64,
    atoms_len: usize,
    hist: [*]Bucket,
    hist_len: usize,
    resolution: f64,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
) void;

pub extern fn launch_shared_memory(
    x_pos: [*]f64,
    y_pos: [*]f64,
    z_pos: [*]f64,
    atoms_len: usize,
    hist: [*]Bucket,
    hist_len: usize,
    resolution: f64,
    grid_size: u32,
    block_size: u32,
    shared_mem_size: usize,
) void;

pub extern fn launch_output_privatization(
    x_pos: [*]f64,
    y_pos: [*]f64,
    z_pos: [*]f64,
    atoms_len: usize,
    hist_2d: [*]Bucket,
    hist_len: usize,
    resolution: f64,
    grid_size: u32,
    block_size: u32,
    shared_mem_size: usize,
) void;

pub extern fn launch_reduction(
    hist_2d: [*]Bucket,
    hist_2d_width: usize,
    hist: [*]Bucket,
    reduction_grid_size: u32,
    reduction_block_size: u32,
    reduction_shared_mem_size: usize,
) void;
