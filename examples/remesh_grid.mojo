"""Minimal public API example."""

from algorithms3d import RemeshOptions, TriangleMesh, Vec3, surface_relax


def main() raises:
    var mesh = TriangleMesh()
    _ = mesh.add_vertex(Vec3(-1.0, -1.0, 0.0))
    _ = mesh.add_vertex(Vec3(1.0, -1.0, 0.0))
    _ = mesh.add_vertex(Vec3(0.8, 1.0, 0.0))
    _ = mesh.add_vertex(Vec3(-0.8, 0.7, 0.0))
    _ = mesh.add_vertex(Vec3(0.1, 0.0, 0.0))
    mesh.add_triangle(0, 1, 4)
    mesh.add_triangle(1, 2, 4)
    mesh.add_triangle(2, 3, 4)
    mesh.add_triangle(3, 0, 4)

    var options = RemeshOptions()
    options.preserve_features = False
    var stats = surface_relax(mesh, options)
    print("quality objective:", stats.initial.objective, "->", stats.final.objective)
