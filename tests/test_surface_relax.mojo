"""Regression and contract tests for the public v0.1 API."""

from std.math import abs, cos, sin

from algorithms3d import (
    RemeshOptions,
    TriangleMesh,
    Vec3,
    compute_quality,
    surface_relax,
    validate_mesh,
)


def assert_true(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def make_irregular_grid(side: Int = 7) -> TriangleMesh:
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


def test_surface_relax() raises:
    var mesh = make_irregular_grid()
    var source = mesh.copy()
    validate_mesh(mesh)
    var before = compute_quality(mesh)
    var options = RemeshOptions()
    options.iterations = 12
    options.relaxation = 0.7
    options.qem_weight = 0.8
    options.preserve_features = False
    var stats = surface_relax(mesh, options)
    var after = compute_quality(mesh)

    assert_true(stats.iterations_completed > 0, "an improvement should commit")
    assert_true(after.objective < before.objective, "objective should improve")
    assert_true(
        after.edge_coefficient_of_variation
        < before.edge_coefficient_of_variation,
        "edge distribution should become more uniform",
    )
    assert_true(
        stats.max_surface_deviation < 1.0e-6,
        "vertices must remain on the source surface",
    )
    for vertex in range(mesh.vertex_count()):
        assert_true(
            abs(mesh.vertices[vertex].z) < 1.0e-6,
            "a planar source must remain planar",
        )
        var x = vertex % 7
        var y = vertex // 7
        if x == 0 or y == 0 or x == 6 or y == 6:
            assert_true(
                (mesh.vertices[vertex] - source.vertices[vertex]).length()
                < 1.0e-6,
                "boundary vertices must remain fixed",
            )


def test_validation_rejects_degenerate_face() raises:
    var mesh = TriangleMesh()
    _ = mesh.add_vertex(Vec3(0.0, 0.0, 0.0))
    _ = mesh.add_vertex(Vec3(1.0, 0.0, 0.0))
    mesh.add_triangle(0, 1, 1)
    var rejected = False
    try:
        validate_mesh(mesh)
    except:
        rejected = True
    assert_true(rejected, "validation must reject repeated triangle corners")


def main() raises:
    test_surface_relax()
    test_validation_rejects_degenerate_face()
    print("PASS: mojo-algorithms-3d public API")
