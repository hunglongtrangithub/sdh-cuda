const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "sdh_cuda",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.root_module.link_libc = true;
    exe.root_module.linkSystemLibrary("cudart", .{});

    // Compile each .cu kernel file with nvcc and link the resulting object
    const cu_sources = [_][]const u8{
        "src/kernels/grid_2d.cu",
        "src/kernels/shared_memory.cu",
        "src/kernels/output_privatization.cu",
    };

    for (cu_sources) |cu_src| {
        const nvcc = b.addSystemCommand(&.{ "nvcc", "-c", "-o" });
        const obj_name = try std.fmt.allocPrint(b.allocator, "{s}.o", .{std.fs.path.stem(cu_src)});
        const obj = nvcc.addOutputFileArg(obj_name);
        nvcc.addFileArg(b.path(cu_src));
        nvcc.addArgs(&.{
            "--compiler-options",
            "-fPIC",
        });
        exe.root_module.addObjectFile(obj);
    }

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_exe_tests.step);
}
