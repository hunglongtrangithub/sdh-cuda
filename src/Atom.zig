const std = @import("std");
const box_size: f64 = 23000.0;

x_pos: [*]f64,
y_pos: [*]f64,
z_pos: [*]f64,
len: usize,

const Self = @This();

pub fn init(
    allocator: std.mem.Allocator,
    len: usize,
    seed: u64,
) std.mem.Allocator.Error!Self {
    const x_pos = try allocator.alloc(f64, len);
    const y_pos = try allocator.alloc(f64, len);
    const z_pos = try allocator.alloc(f64, len);

    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    for (0..len) |i| {
        x_pos[i] = rand.float(f64) * box_size;
        y_pos[i] = rand.float(f64) * box_size;
        z_pos[i] = rand.float(f64) * box_size;
    }

    return Self{
        .x_pos = x_pos.ptr,
        .y_pos = y_pos.ptr,
        .z_pos = z_pos.ptr,
        .len = len,
    };
}

pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
    allocator.free(self.x_pos[0..self.len]);
    allocator.free(self.y_pos[0..self.len]);
    allocator.free(self.z_pos[0..self.len]);
}
