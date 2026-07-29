"""Deterministic structural cleanup for indexed triangle meshes."""

from std.collections import Dict, List

from .mesh import TriangleMesh, Vec3, validate_mesh


struct RepairOptions(
    Copyable, Movable, ImplicitlyCopyable, ImplicitlyDestructible
):
    """Controls for the conservative cleanup pass."""

    var degenerate_epsilon: Float32
    var remove_invalid_faces: Bool
    var remove_duplicate_faces: Bool
    var compact_unreferenced_vertices: Bool

    def __init__(out self):
        self.degenerate_epsilon = 1.0e-12
        self.remove_invalid_faces = True
        self.remove_duplicate_faces = True
        self.compact_unreferenced_vertices = True


struct RepairStats(
    Copyable, Movable, ImplicitlyCopyable, ImplicitlyDestructible
):
    """Counts describing exactly what a cleanup pass changed."""

    var input_vertices: Int
    var input_faces: Int
    var output_vertices: Int
    var output_faces: Int
    var removed_invalid_faces: Int
    var removed_degenerate_faces: Int
    var removed_duplicate_faces: Int
    var removed_unreferenced_vertices: Int

    def __init__(out self):
        self.input_vertices = 0
        self.input_faces = 0
        self.output_vertices = 0
        self.output_faces = 0
        self.removed_invalid_faces = 0
        self.removed_degenerate_faces = 0
        self.removed_duplicate_faces = 0
        self.removed_unreferenced_vertices = 0

    def changed(self) -> Bool:
        return (
            self.removed_invalid_faces > 0
            or self.removed_degenerate_faces > 0
            or self.removed_duplicate_faces > 0
            or self.removed_unreferenced_vertices > 0
        )


def _indices_in_range(mesh: TriangleMesh, a: Int, b: Int, c: Int) -> Bool:
    return (
        a >= 0
        and b >= 0
        and c >= 0
        and a < mesh.vertex_count()
        and b < mesh.vertex_count()
        and c < mesh.vertex_count()
    )


def _face_key(a: Int, b: Int, c: Int, vertex_count: Int) -> Int:
    var low = min(a, min(b, c))
    var high = max(a, max(b, c))
    var middle = a + b + c - low - high
    return (low * vertex_count + middle) * vertex_count + high


def _compact_vertices(mut mesh: TriangleMesh) -> Int:
    var referenced = List[Bool]()
    var remap = List[Int]()
    for _ in range(mesh.vertex_count()):
        referenced.append(False)
        remap.append(-1)
    for index in mesh.indices:
        referenced[index] = True

    var compacted = List[Vec3]()
    for index in range(mesh.vertex_count()):
        if referenced[index]:
            remap[index] = len(compacted)
            compacted.append(mesh.vertices[index])
    for index in range(len(mesh.indices)):
        mesh.indices[index] = remap[mesh.indices[index]]

    var removed = mesh.vertex_count() - len(compacted)
    mesh.vertices = compacted^
    return removed


def repair_mesh(
    mut mesh: TriangleMesh, options: RepairOptions = RepairOptions()
) raises -> RepairStats:
    """Remove unusable faces and compact vertices without changing valid shape.

    Duplicate detection treats triangles with the same three vertex indices as
    duplicates even if their winding differs. The first occurrence is kept.
    """
    if options.degenerate_epsilon < 0.0:
        raise Error("degenerate_epsilon cannot be negative")
    if len(mesh.indices) % 3 != 0 and not options.remove_invalid_faces:
        raise Error("partial triangle data requires remove_invalid_faces")

    var stats = RepairStats()
    stats.input_vertices = mesh.vertex_count()
    stats.input_faces = len(mesh.indices) // 3
    var kept = List[Int]()
    var seen_faces = Dict[Int, Bool]()

    for face in range(len(mesh.indices) // 3):
        var base = face * 3
        var a = mesh.indices[base]
        var b = mesh.indices[base + 1]
        var c = mesh.indices[base + 2]

        if not _indices_in_range(mesh, a, b, c):
            if options.remove_invalid_faces:
                stats.removed_invalid_faces += 1
                continue
            raise Error("triangle index is outside the vertex buffer")

        if a == b or b == c or c == a:
            stats.removed_degenerate_faces += 1
            continue
        var twice_area = (
            (mesh.vertices[b] - mesh.vertices[a])
            .cross(mesh.vertices[c] - mesh.vertices[a])
            .length()
        )
        if twice_area <= options.degenerate_epsilon:
            stats.removed_degenerate_faces += 1
            continue
        if options.remove_duplicate_faces:
            var face_key = _face_key(a, b, c, mesh.vertex_count())
            if face_key in seen_faces:
                stats.removed_duplicate_faces += 1
                continue
            seen_faces[face_key] = True
        kept.append(a)
        kept.append(b)
        kept.append(c)

    if len(mesh.indices) % 3 != 0:
        stats.removed_invalid_faces += 1
    mesh.indices = kept^

    if options.compact_unreferenced_vertices:
        stats.removed_unreferenced_vertices = _compact_vertices(mesh)
    stats.output_vertices = mesh.vertex_count()
    stats.output_faces = mesh.triangle_count()
    if stats.output_faces > 0:
        validate_mesh(mesh)
    return stats
