const std = @import("std");

const box_size: f64 = 23000.0;

pub const Bucket = struct {
    d_cnt: u64,
};

arr: []Bucket,
resolution: f64,

const Self = @This();

pub fn init(allocator: std.mem.Allocator, resolution: f64) std.mem.Allocator.Error!Self {
    const n = numBuckets(resolution);
    const arr = try allocator.alloc(Bucket, n);
    var self = Self{ .arr = arr, .resolution = resolution };
    self.reset();
    return self;
}

pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
    allocator.free(self.arr);
}

pub fn numBuckets(resolution: f64) usize {
    return @as(usize, @intFromFloat(box_size * @sqrt(3.0) / resolution)) + 1;
}

pub fn reset(self: *Self) void {
    for (self.arr) |*b| {
        b.d_cnt = 0;
    }
}

fn countDigits(n: usize) usize {
    if (n == 0) return 1;
    var count: usize = 0;
    var v = n;
    while (v > 0) : (v /= 10) count += 1;
    return count;
}

pub fn display(self: *const Self, writer: *std.Io.Writer) !void {
    var total_cnt: u64 = 0;
    const width = countDigits(self.arr.len - 1);

    for (self.arr, 0..) |b, i| {
        if (i % 5 == 0) {
            try writer.writeAll("\n");
            // Zero-padded index
            var idx_buf: [countDigits(std.math.maxInt(usize))]u8 = undefined;
            var pos: usize = idx_buf.len;
            var v: usize = i;
            if (v == 0) {
                pos -= 1;
                idx_buf[pos] = '0';
            } else while (v > 0) {
                pos -= 1;
                idx_buf[pos] = @intCast('0' + v % 10);
                v /= 10;
            }
            const digits = idx_buf.len - pos;
            var pad = width -| digits;
            while (pad > 0) : (pad -= 1) try writer.writeAll("0");
            try writer.writeAll(idx_buf[pos..]);
            try writer.writeAll(": ");
        }

        try writer.print("{d:>15} ", .{b.d_cnt});
        total_cnt += b.d_cnt;

        if (i == self.arr.len - 1)
            try writer.print("\n T:{d} \n", .{total_cnt})
        else
            try writer.print("| ", .{});
    }
}
