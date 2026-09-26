"""Small, owned triangle-mesh model shared by the public algorithms."""

from std.collections import List
from std.math import sqrt


struct Vec3(Copyable, Movable, ImplicitlyCopyable):
    """A compact Float32 three-vector."""

    var x: Float32
    var y: Float32
    var z: Float32

    def __init__(out self, x: Float32, y: Float32, z: Float32):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def zero() -> Vec3:
        return Vec3(0.0, 0.0, 0.0)

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: Float32) -> Vec3:
        return Vec3(self.x * scale, self.y * scale, self.z * scale)

    def dot(self, other: Vec3) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> Float32:
        return sqrt(self.dot(self))

    def normalized(self) -> Vec3:
        var magnitude = self.length()
        if magnitude <= 1.0e-12:
            return Vec3.zero()
        return self * (1.0 / magnitude)


struct TriangleMesh(Movable):
    """Owned vertices and a flat, counter-clockwise triangle index buffer."""

    var vertices: List[Vec3]
    var indices: List[Int]

    def __init__(out self):
        self.vertices = List[Vec3]()
        self.indices = List[Int]()

    def add_vertex(mut self, point: Vec3) -> Int:
        self.vertices.append(point)
        return len(self.vertices) - 1

    def add_triangle(mut self, a: Int, b: Int, c: Int):
        self.indices.append(a)
        self.indices.append(b)
        self.indices.append(c)

    def vertex_count(self) -> Int:
        return len(self.vertices)

    def triangle_count(self) -> Int:
        return len(self.indices) // 3

    def copy(self) -> TriangleMesh:
        var result = TriangleMesh()
        for vertex in self.vertices:
            result.vertices.append(vertex)
        for index in self.indices:
            result.indices.append(index)
        return result^


struct MeshQuality(Copyable, Movable, ImplicitlyCopyable):
    """Scale-independent triangle and edge-distribution metrics."""

    var mean_edge_length: Float32
    var edge_coefficient_of_variation: Float32
    var mean_triangle_quality: Float32
    var min_triangle_quality: Float32
    var objective: Float32

    def __init__(out self):
        self.mean_edge_length = 0.0
        self.edge_coefficient_of_variation = 0.0
        self.mean_triangle_quality = 0.0
        self.min_triangle_quality = 1.0
        self.objective = 0.0


def validate_mesh(mesh: TriangleMesh) raises:
    """Reject malformed indices, repeated corners, and zero-area faces."""
    if len(mesh.indices) % 3 != 0:
        raise Error("triangle index buffer length must be divisible by three")
    for face in range(mesh.triangle_count()):
        var base = face * 3
        var a = mesh.indices[base]
        var b = mesh.indices[base + 1]
        var c = mesh.indices[base + 2]
        if a < 0 or b < 0 or c < 0:
            raise Error("triangle index cannot be negative")
        if a >= mesh.vertex_count() or b >= mesh.vertex_count() or c >= mesh.vertex_count():
            raise Error("triangle index is outside the vertex buffer")
        if a == b or b == c or c == a:
            raise Error("triangle contains a repeated vertex")
        if (
            (mesh.vertices[b] - mesh.vertices[a])
            .cross(mesh.vertices[c] - mesh.vertices[a])
            .length()
            <= 1.0e-12
        ):
            raise Error("triangle has zero area")


def compute_quality(mesh: TriangleMesh) -> MeshQuality:
    """Compute edge uniformity and normalized equilateral-triangle quality."""
    var result = MeshQuality()
    var edge_sum: Float32 = 0.0
    var edge_square_sum: Float32 = 0.0
    var edge_count = 0
    var quality_sum: Float32 = 0.0
    for face in range(mesh.triangle_count()):
        var base = face * 3
        var a = mesh.vertices[mesh.indices[base]]
        var b = mesh.vertices[mesh.indices[base + 1]]
        var c = mesh.vertices[mesh.indices[base + 2]]
        var ab = (a - b).length()
        var bc = (b - c).length()
        var ca = (c - a).length()
        for edge in [ab, bc, ca]:
            edge_sum += edge
            edge_square_sum += edge * edge
            edge_count += 1
        var denominator = ab * ab + bc * bc + ca * ca
        var twice_area = (b - a).cross(c - a).length()
        var quality = (
            2.0 * sqrt(Float32(3.0)) * twice_area / denominator
            if denominator > 1.0e-12
            else 0.0
        )
        quality_sum += quality
        if quality < result.min_triangle_quality:
            result.min_triangle_quality = quality
    if edge_count > 0:
        result.mean_edge_length = edge_sum / Float32(edge_count)
        var variance = (
            edge_square_sum / Float32(edge_count)
            - result.mean_edge_length * result.mean_edge_length
        )
        if variance < 0.0:
            variance = 0.0
        if result.mean_edge_length > 1.0e-12:
            result.edge_coefficient_of_variation = (
                sqrt(variance) / result.mean_edge_length
            )
    if mesh.triangle_count() > 0:
        result.mean_triangle_quality = quality_sum / Float32(mesh.triangle_count())
    result.objective = (
        result.edge_coefficient_of_variation
        + 0.35 * (1.0 - result.mean_triangle_quality)
    )
    return result
