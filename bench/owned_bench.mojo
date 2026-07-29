"""Timed owned-mesh kernels used by the Python comparison harness."""

from std.math import cos, sin
from std.time import perf_counter_ns

from algorithms3d import (
    RemeshOptions,
    TriangleMesh,
    Vec3,
    compute_quality,
    repair_mesh,
    surface_relax,
)


def make_grid(side: Int) -> TriangleMesh:
    var mesh = TriangleMesh()
    for y in range(side):
        for x in range(side):
            var boundary = (
                x == 0 or y == 0 or x == side - 1 or y == side - 1
            )
            var index = y * side + x
            var jitter_x = (
                Float32(0.0)
                if boundary
                else sin(Float32(index) * 2.17) * 0.095
            )
            var jitter_y = (
                Float32(0.0)
                if boundary
                else cos(Float32(index) * 1.31) * 0.075
            )
            _ = mesh.add_vertex(
                Vec3(
                    -1.0 + 2.0 * Float32(x) / Float32(side - 1) + jitter_x,
                    -1.0 + 2.0 * Float32(y) / Float32(side - 1) + jitter_y,
                    0.0,
                )
            )
    for y in range(side - 1):
        for x in range(side - 1):
            var a = y * side + x
            var b = a + 1
            var c = (y + 1) * side + x
            var d = c + 1
            mesh.add_triangle(a, b, d)
            mesh.add_triangle(a, d, c)
    return mesh^


def make_duplicate_soup(unique_faces: Int) -> TriangleMesh:
    var mesh = TriangleMesh()
    for face in range(unique_faces):
        var x = Float32(face)
        var a = mesh.add_vertex(Vec3(x, 0.0, 0.0))
        var b = mesh.add_vertex(Vec3(x + 0.5, 1.0, 0.0))
        var c = mesh.add_vertex(Vec3(x + 1.0, 0.0, 0.0))
        mesh.add_triangle(a, b, c)
        mesh.add_triangle(c, b, a)
    return mesh^


def main() raises:
    var quality_mesh = make_grid(200)
    var quality_guard: Float32 = 0.0
    for repeat in range(8):
        var start = perf_counter_ns()
        var quality = compute_quality(quality_mesh)
        var elapsed = perf_counter_ns() - start
        quality_guard += quality.objective
        if repeat > 0:
            print("quality_ns", elapsed)

    var repair_guard = 0
    for repeat in range(8):
        var repair_mesh_input = make_duplicate_soup(5_000)
        var start = perf_counter_ns()
        var repair_stats = repair_mesh(repair_mesh_input)
        var elapsed = perf_counter_ns() - start
        repair_guard += repair_stats.output_faces
        if repeat > 0:
            print("repair_ns", elapsed)

    var relax_source = make_grid(12)
    var options = RemeshOptions()
    options.iterations = 2
    options.relaxation = 0.7
    options.qem_weight = 0.8
    options.preserve_features = False
    var relax_guard = 0
    for repeat in range(6):
        var relax_mesh = relax_source.copy()
        var start = perf_counter_ns()
        var relax_stats = surface_relax(relax_mesh, options)
        var elapsed = perf_counter_ns() - start
        relax_guard += relax_stats.iterations_completed
        if repeat > 0:
            print("relax_ns", elapsed)

    print("guard", quality_guard, repair_guard, relax_guard)
