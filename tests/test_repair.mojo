"""Contract tests for conservative triangle-mesh cleanup."""

from algorithms3d import (
    RepairOptions,
    TriangleMesh,
    Vec3,
    repair_mesh,
    validate_mesh,
)


def assert_true(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def test_mixed_damage_cleanup() raises:
    var mesh = TriangleMesh()
    _ = mesh.add_vertex(Vec3(0.0, 0.0, 0.0))
    _ = mesh.add_vertex(Vec3(1.0, 0.0, 0.0))
    _ = mesh.add_vertex(Vec3(0.0, 1.0, 0.0))
    _ = mesh.add_vertex(Vec3(4.0, 4.0, 4.0))  # Unreferenced.
    mesh.add_triangle(0, 1, 2)  # Valid.
    mesh.add_triangle(2, 1, 0)  # Duplicate with reversed winding.
    mesh.add_triangle(0, 1, 1)  # Repeated corner.
    mesh.add_triangle(0, 1, 9)  # Out of range.

    var stats = repair_mesh(mesh)
    validate_mesh(mesh)
    assert_true(stats.changed(), "damaged input should report changes")
    assert_true(stats.removed_invalid_faces == 1, "invalid face count")
    assert_true(stats.removed_degenerate_faces == 1, "degenerate face count")
    assert_true(stats.removed_duplicate_faces == 1, "duplicate face count")
    assert_true(
        stats.removed_unreferenced_vertices == 1,
        "unreferenced vertex count",
    )
    assert_true(mesh.vertex_count() == 3, "three referenced vertices remain")
    assert_true(mesh.triangle_count() == 1, "one valid triangle remains")
    assert_true(
        mesh.indices[0] == 0
        and mesh.indices[1] == 1
        and mesh.indices[2] == 2,
        "the first valid triangle and its winding must be preserved",
    )


def test_valid_mesh_is_stable() raises:
    var mesh = TriangleMesh()
    _ = mesh.add_vertex(Vec3(0.0, 0.0, 0.0))
    _ = mesh.add_vertex(Vec3(1.0, 0.0, 0.0))
    _ = mesh.add_vertex(Vec3(0.0, 1.0, 0.0))
    mesh.add_triangle(0, 1, 2)
    var stats = repair_mesh(mesh, RepairOptions())
    assert_true(not stats.changed(), "valid compact mesh should be unchanged")


def main() raises:
    test_mixed_damage_cleanup()
    test_valid_mesh_is_stable()
    print("PASS: mesh repair cleanup")
