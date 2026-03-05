const std = @import("std");
const computation = @import("computation.zig");
const Atom = @import("Atom.zig");
const Histogram = @import("Histogram.zig");

const print = std.debug.print;

fn nsToMsF32(ns: u64) f32 {
    return @as(f32, @floatFromInt(ns)) / @as(f32, @floatFromInt(std.time.ns_per_ms));
}

const GpuAlgorithmEntry = struct {
    name: []const u8,
    algo: computation.KernelAlgorithm,
};

const gpu_algorithms = [_]GpuAlgorithmEntry{
    .{ .name = "GPU 2D grid (baseline)", .algo = .grid_2d },
    .{ .name = "GPU shared memory", .algo = .shared_memory },
    .{ .name = "GPU output privatization", .algo = .output_privatization },
};

fn displayHist(hist: *const Histogram) !void {
    var buf: [4096]u8 = undefined;
    var w = std.fs.File.stdout().writer(&buf);
    try hist.display(&w.interface);
    try w.interface.flush();
}

fn runGpuVersion(
    allocator: std.mem.Allocator,
    atoms: *const Atom,
    ref_hist: *const Histogram,
    ref_time_ms: f32,
    ref_label: []const u8,
    version_name: []const u8,
    algorithm: computation.KernelAlgorithm,
    resolution: f64,
    block_size: usize,
) !f32 {
    var hist = try Histogram.init(allocator, resolution);
    defer hist.deinit(allocator);

    print("========================================\n", .{});
    print("Running {s} version\n", .{version_name});

    const time_gpu_ms = computation.timeAndFillHistogramGpu(
        atoms,
        &hist,
        block_size,
        algorithm,
    ) catch |e| {
        print("Error running {s} version. Exiting\n", .{version_name});
        return e;
    };

    print("{s} version histogram:\n", .{version_name});
    try displayHist(&hist);
    print("{s} time in milliseconds: {d:.3}\n", .{ version_name, time_gpu_ms });

    // Compute and display diff against reference
    const n = ref_hist.arr.len;
    const diff_buckets = try allocator.alloc(Histogram.Bucket, n);
    defer allocator.free(diff_buckets);
    for (0..n) |i| {
        diff_buckets[i].d_cnt = hist.arr[i].d_cnt -% ref_hist.arr[i].d_cnt;
    }
    var diff_hist = Histogram{ .arr = diff_buckets, .resolution = resolution };
    print("{s} version histogram diff:\n", .{version_name});
    try displayHist(&diff_hist);

    if (time_gpu_ms > 0) {
        const speedup = ref_time_ms / time_gpu_ms;
        print("Speedup ({s} vs {s}): {d:.2}x\n", .{ version_name, ref_label, speedup });
    }
    print("========================================\n", .{});

    return time_gpu_ms;
}

fn runExperiments(allocator: std.mem.Allocator, csv: *std.io.Writer) !void {
    const num_atoms_list = [_]usize{ 10000, 50000, 100000 };
    const resolution_list = [_]f64{ 100, 200, 500 };
    const block_sizes = [_]usize{ 32, 64, 128, 256 };

    try csv.writeAll("run_id,num_atoms,resolution,block_size,algorithm,time_ms,speedup\n");

    var run_id: u32 = 0;

    for (num_atoms_list) |num_atoms| {
        for (resolution_list) |resolution| {
            for (block_sizes) |block_size| {
                run_id += 1;

                print(
                    "Running configuration {d}: atoms={d}, resolution={d:.1}, block_size={d}\n",
                    .{ run_id, num_atoms, resolution, block_size },
                );

                const atoms = try Atom.init(allocator, num_atoms, 0);
                defer atoms.deinit(allocator);

                var hist = try Histogram.init(allocator, resolution);
                defer hist.deinit(allocator);

                // CPU run
                var time_cpu_ms: f32 = 0;
                if (computation.timeAndFillHistogramCpu(&atoms, &hist)) |ns| {
                    time_cpu_ms = nsToMsF32(ns);
                    try csv.print(
                        "{d},{d},{d:.1},{d},CPU,{d:.3},1.0\n",
                        .{ run_id, num_atoms, resolution, block_size, time_cpu_ms },
                    );
                    print("  CPU time: {d:.3} ms\n", .{time_cpu_ms});
                } else |_| {
                    print("  CPU calculation failed\n", .{});
                }

                // GPU runs
                const exp_algorithms = [_]struct { name: []const u8, algo: computation.KernelAlgorithm }{
                    .{ .name = "GRID_2D", .algo = .grid_2d },
                    .{ .name = "SHARED_MEM", .algo = .shared_memory },
                    .{ .name = "OUTPUT_PRIV", .algo = .output_privatization },
                };

                for (exp_algorithms) |entry| {
                    hist.reset();

                    if (computation.timeAndFillHistogramGpu(
                        &atoms,
                        &hist,
                        block_size,
                        entry.algo,
                    )) |time_gpu_ms| {
                        const speedup = if (time_gpu_ms > 0) time_cpu_ms / time_gpu_ms else 0;
                        try csv.print(
                            "{d},{d},{d:.1},{d},{s},{d:.3},{d:.2}\n",
                            .{ run_id, num_atoms, resolution, block_size, entry.name, time_gpu_ms, speedup },
                        );
                        print(
                            "  {s} time: {d:.3} ms (speedup: {d:.2}x)\n",
                            .{ entry.name, time_gpu_ms, speedup },
                        );
                    } else |_| {
                        print("  {s} calculation failed\n", .{entry.name});
                    }
                }
            }
        }
    }
}

fn experiment(allocator: std.mem.Allocator, csv_path: []const u8) !void {
    print("Running experiments...\n", .{});
    print("Creating CSV file {s}\n", .{csv_path});

    const file = std.fs.cwd().createFile(csv_path, .{}) catch |err| {
        print("Failed to create CSV file: {any}\n", .{err});
        return err;
    };
    defer file.close();

    var csv_buffer: [1024]u8 = undefined;
    var csv_file_writer = file.writer(&csv_buffer);
    const csv = &csv_file_writer.interface;
    try runExperiments(allocator, csv);
    try csv.flush();
    print("Experiments completed. Results are saved in {s}.\n", .{csv_path});
}

fn demo(
    allocator: std.mem.Allocator,
    particle_count: usize,
    resolution: f64,
    block_size: usize,
    gpu_only: bool,
) !void {
    const atoms = try Atom.init(allocator, particle_count, 0);
    defer atoms.deinit(allocator);

    if (gpu_only) {
        // GPU-only mode: run grid_2d as reference, compare others against it
        const baseline = gpu_algorithms[0].name;
        print("GPU-only mode: using {s} as baseline\n", .{baseline});

        var ref_hist = try Histogram.init(allocator, resolution);
        defer ref_hist.deinit(allocator);

        const ref_time_ms = runGpuVersion(
            allocator,
            &atoms,
            &ref_hist,
            0,
            "",
            gpu_algorithms[0].name,
            gpu_algorithms[0].algo,
            resolution,
            block_size,
        ) catch {
            print("{s} failed, cannot establish GPU baseline\n", .{gpu_algorithms[0].name});
            return;
        };

        print("{s} time in milliseconds: {d:.3} (baseline)\n", .{ gpu_algorithms[0].name, ref_time_ms });

        // Run remaining GPU algorithms compared against the baseline
        for (gpu_algorithms[1..]) |entry| {
            _ = try runGpuVersion(
                allocator,
                &atoms,
                &ref_hist,
                ref_time_ms,
                gpu_algorithms[0].name,
                entry.name,
                entry.algo,
                resolution,
                block_size,
            );
        }
    } else {
        // Default mode: run CPU first, compare all GPU versions against CPU
        var hist_cpu = try Histogram.init(allocator, resolution);
        defer hist_cpu.deinit(allocator);

        print("Running CPU version...\n", .{});
        const time_cpu_ns = computation.timeAndFillHistogramCpu(&atoms, &hist_cpu) catch {
            print("CPU histogram computation failed\n", .{});
            return;
        };
        const time_cpu_ms = nsToMsF32(time_cpu_ns);
        print("CPU time in milliseconds: {d:.3}\n", .{time_cpu_ms});

        for (gpu_algorithms) |entry| {
            _ = try runGpuVersion(
                allocator,
                &atoms,
                &hist_cpu,
                time_cpu_ms,
                "CPU",
                entry.name,
                entry.algo,
                resolution,
                block_size,
            );
        }
    }
}

// Usage / arg parsing

const usage =
    \\Usage: {s} <command> [options]
    \\  demo <num_particles> <bucket_width> <block_size> [--gpu-only]
    \\  experiment [csv_path]
    \\
;

fn parseUsizePos(s: []const u8, name: []const u8) u64 {
    const v = std.fmt.parseInt(usize, s, 10) catch {
        print("Invalid {s}: '{s}'\n", .{ name, s });
        std.process.exit(1);
    };
    if (v == 0) {
        print("{s} must be greater than 0\n", .{name});
        std.process.exit(1);
    }
    return v;
}

fn parseF64Pos(s: []const u8, name: []const u8) f64 {
    const v = std.fmt.parseFloat(f64, s) catch {
        print("Invalid {s}: '{s}'\n", .{ name, s });
        std.process.exit(1);
    };
    if (v <= 0.0) {
        print("{s} must be positive\n", .{name});
        std.process.exit(1);
    }
    return v;
}

fn nextArgOrUsage(
    args: *std.process.ArgIterator,
    exe_cmd: []const u8,
    name: []const u8,
) []const u8 {
    return args.next() orelse {
        print("{s} needed.\n", .{name});
        print(usage, .{exe_cmd});
        std.process.exit(1);
    };
}

const Command = union(enum) {
    demo: struct {
        num_particles: usize,
        bucket_width: f64,
        block_size: usize,
        gpu_only: bool,
    },
    experiment: []const u8,
};

fn parseArgs(args: *std.process.ArgIterator) Command {
    const exe_cmd = std.fs.path.basename(args.next() orelse std.process.exit(0));

    const sub_cmd = args.next() orelse {
        print("Spatial Distance Histogram Computation on CUDA\n", .{});
        print(usage, .{exe_cmd});
        std.process.exit(0);
    };

    if (std.mem.eql(u8, sub_cmd, "experiment")) {
        const csv_path = args.next() orelse "experiment_results.csv";
        return .{ .experiment = csv_path };
    } else if (std.mem.eql(u8, sub_cmd, "demo")) {
        const num_particles_str = nextArgOrUsage(args, exe_cmd, "Number of particles");
        const bucket_width_str = nextArgOrUsage(args, exe_cmd, "Bucket width");
        const block_size_str = nextArgOrUsage(args, exe_cmd, "Block size");

        const num_particles = parseUsizePos(num_particles_str, "particle count");
        const bucket_width = parseF64Pos(bucket_width_str, "bucket width");
        const block_size = parseUsizePos(block_size_str, "block size");

        var gpu_only = false;
        if (args.next()) |flag| {
            if (std.mem.eql(u8, flag, "--gpu-only")) {
                gpu_only = true;
            } else {
                print("Unknown flag: {s}\n", .{flag});
                print(usage, .{exe_cmd});
                std.process.exit(1);
            }
        }

        return .{ .demo = .{
            .num_particles = num_particles,
            .bucket_width = bucket_width,
            .block_size = block_size,
            .gpu_only = gpu_only,
        } };
    } else {
        print("Unknown command: {s}\n", .{sub_cmd});
        print(usage, .{exe_cmd});
        std.process.exit(1);
    }
}

// Entry point

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    const cmd = parseArgs(&args);

    switch (cmd) {
        .demo => |d| try demo(
            allocator,
            d.num_particles,
            d.bucket_width,
            d.block_size,
            d.gpu_only,
        ),
        .experiment => |csv_path| try experiment(
            allocator,
            csv_path,
        ),
    }
}
