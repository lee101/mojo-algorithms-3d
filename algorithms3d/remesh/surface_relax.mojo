"""Surface Voronoi relaxation constrained by per-vertex error quadrics."""

from std.collections import Dict, List
from std.math import abs, cos

from ..mesh import MeshQuality, TriangleMesh, Vec3, compute_quality, validate_mesh


struct RemeshOptions(Copyable, Movable, ImplicitlyCopyable, ImplicitlyDestructible):
    """Controls for deterministic, topology-preserving surface relaxation."""

    var iterations: Int
    var relaxation: Float32
    var qem_weight: Float32
    var preserve_boundaries: Bool
    var preserve_features: Bool
    var feature_angle_degrees: Float32
    var convergence: Float32

    def __init__(out self):
        self.iterations = 8
        self.relaxation = 0.55
        self.qem_weight = 0.75
        self.preserve_boundaries = True
        self.preserve_features = True
        self.feature_angle_degrees = 50.0
        self.convergence = 1.0e-5


struct RemeshStats(Copyable, Movable, ImplicitlyCopyable, ImplicitlyDestructible):
    """Measured result of a remeshing pass."""

    var iterations_requested: Int
    var iterations_completed: Int
    var converged: Bool
    var locked_boundary_vertices: Int
    var locked_feature_vertices: Int
    var max_surface_deviation: Float32
    var initial: MeshQuality
    var final: MeshQuality

    def __init__(out self):
        self.iterations_requested = 0
        self.iterations_completed = 0
        self.converged = False
        self.locked_boundary_vertices = 0
        self.locked_feature_vertices = 0
        self.max_surface_deviation = 0.0
        self.initial = MeshQuality()
        self.final = MeshQuality()

    def quality_improvement(self) -> Float32:
        if self.initial.objective <= 1.0e-12:
            return 0.0
        return (
            (self.initial.objective - self.final.objective)
            / self.initial.objective
        )


struct _Quadric(Copyable, Movable, ImplicitlyCopyable, ImplicitlyDestructible):
    var q00: Float32
    var q01: Float32
    var q02: Float32
    var q03: Float32
    var q11: Float32
    var q12: Float32
    var q13: Float32
    var q22: Float32
    var q23: Float32

    def __init__(out self):
        self.q00 = 0.0
        self.q01 = 0.0
        self.q02 = 0.0
        self.q03 = 0.0
        self.q11 = 0.0
        self.q12 = 0.0
        self.q13 = 0.0
        self.q22 = 0.0
        self.q23 = 0.0

    def add_plane(mut self, normal: Vec3, d: Float32, weight: Float32):
        self.q00 += normal.x * normal.x * weight
        self.q01 += normal.x * normal.y * weight
        self.q02 += normal.x * normal.z * weight
        self.q03 += normal.x * d * weight
        self.q11 += normal.y * normal.y * weight
        self.q12 += normal.y * normal.z * weight
        self.q13 += normal.y * d * weight
        self.q22 += normal.z * normal.z * weight
        self.q23 += normal.z * d * weight


struct _VertexFaces(Movable):
    var offsets: List[Int]
    var faces: List[Int]

    def __init__(out self):
        self.offsets = List[Int]()
        self.faces = List[Int]()


def _build_vertex_faces(mesh: TriangleMesh) -> _VertexFaces:
    var counts = List[Int]()
    counts.reserve(mesh.vertex_count())
    for _ in range(mesh.vertex_count()):
        counts.append(0)
    for index in mesh.indices:
        counts[index] += 1

    var result = _VertexFaces()
    result.offsets.reserve(mesh.vertex_count() + 1)
    result.offsets.append(0)
    for count in counts:
        result.offsets.append(result.offsets[len(result.offsets) - 1] + count)
    result.faces.reserve(len(mesh.indices))
    for _ in mesh.indices:
        result.faces.append(0)

    var cursor = List[Int]()
    cursor.reserve(mesh.vertex_count())
    for vertex in range(mesh.vertex_count()):
        cursor.append(result.offsets[vertex])
    for face in range(mesh.triangle_count()):
        var base = face * 3
        for corner in range(3):
            var vertex = mesh.indices[base + corner]
            result.faces[cursor[vertex]] = face
            cursor[vertex] += 1
    return result^


def _face_normal(mesh: TriangleMesh, face: Int) -> Vec3:
    var base = face * 3
    var a = mesh.vertices[mesh.indices[base]]
    var b = mesh.vertices[mesh.indices[base + 1]]
    var c = mesh.vertices[mesh.indices[base + 2]]
    return (b - a).cross(c - a).normalized()


def _face_area(mesh: TriangleMesh, face: Int) -> Float32:
    var base = face * 3
    var a = mesh.vertices[mesh.indices[base]]
    var b = mesh.vertices[mesh.indices[base + 1]]
    var c = mesh.vertices[mesh.indices[base + 2]]
    return (b - a).cross(c - a).length() * 0.5


def _face_centroid(mesh: TriangleMesh, face: Int) -> Vec3:
    var base = face * 3
    return (
        mesh.vertices[mesh.indices[base]]
        + mesh.vertices[mesh.indices[base + 1]]
        + mesh.vertices[mesh.indices[base + 2]]
    ) * (1.0 / 3.0)


def _edge_key(a: Int, b: Int, vertex_count: Int) -> Int:
    return min(a, b) * vertex_count + max(a, b)


def _boundary_vertices(mesh: TriangleMesh) raises -> List[Bool]:
    var edge_counts = Dict[Int, Int]()
    for face in range(mesh.triangle_count()):
        var base = face * 3
        for corner in range(3):
            var a = mesh.indices[base + corner]
            var b = mesh.indices[base + (corner + 1) % 3]
            var key = _edge_key(a, b, mesh.vertex_count())
            if key in edge_counts:
                edge_counts[key] += 1
            else:
                edge_counts[key] = 1

    var boundaries = List[Bool]()
    boundaries.reserve(mesh.vertex_count())
    for _ in range(mesh.vertex_count()):
        boundaries.append(False)
    for face in range(mesh.triangle_count()):
        var base = face * 3
        for corner in range(3):
            var a = mesh.indices[base + corner]
            var b = mesh.indices[base + (corner + 1) % 3]
            if edge_counts[_edge_key(a, b, mesh.vertex_count())] == 1:
                boundaries[a] = True
                boundaries[b] = True
    return boundaries^


def _is_feature_vertex(
    vertex_faces: _VertexFaces,
    face_normals: List[Vec3],
    vertex: Int,
    angle_degrees: Float32,
) -> Bool:
    var cosine_threshold = cos(angle_degrees * 3.141592653589793 / 180.0)
    var begin = vertex_faces.offsets[vertex]
    var end = vertex_faces.offsets[vertex + 1]
    for first_position in range(begin, end):
        var first = vertex_faces.faces[first_position]
        for second_position in range(first_position + 1, end):
            var second = vertex_faces.faces[second_position]
            if face_normals[first].dot(face_normals[second]) < cosine_threshold:
                return True
    return False


def _closest_point_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    var ab = b - a
    var ac = c - a
    var ap = point - a
    var d1 = ab.dot(ap)
    var d2 = ac.dot(ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    var bp = point - b
    var d3 = ab.dot(bp)
    var d4 = ac.dot(bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    var vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + ab * (d1 / (d1 - d3))
    var cp = point - c
    var d5 = ab.dot(cp)
    var d6 = ac.dot(cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    var vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + ac * (d2 / (d2 - d6))
    var va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        return b + (c - b) * (
            (d4 - d3) / ((d4 - d3) + (d5 - d6))
        )
    var denominator = 1.0 / (va + vb + vc)
    return a + ab * (vb * denominator) + ac * (vc * denominator)


def _closest_point_on_source(
    point: Vec3,
    vertex: Int,
    source: TriangleMesh,
    vertex_faces: _VertexFaces,
) -> Vec3:
    var best = source.vertices[vertex]
    var best_distance = Float32(3.4028235e38)
    for position in range(
        vertex_faces.offsets[vertex], vertex_faces.offsets[vertex + 1]
    ):
        var face = vertex_faces.faces[position]
        var base = face * 3
        var candidate = _closest_point_triangle(
            point,
            source.vertices[source.indices[base]],
            source.vertices[source.indices[base + 1]],
            source.vertices[source.indices[base + 2]],
        )
        var delta = candidate - point
        var distance = delta.dot(delta)
        if distance < best_distance:
            best = candidate
            best_distance = distance
    return best


def _solve_qem_target(
    target: Vec3, q: _Quadric, area: Float32, qem_weight: Float32
) -> Vec3:
    var weight = qem_weight / area if area > 1.0e-12 else 0.0
    var a00 = 1.0 + weight * q.q00
    var a01 = weight * q.q01
    var a02 = weight * q.q02
    var a11 = 1.0 + weight * q.q11
    var a12 = weight * q.q12
    var a22 = 1.0 + weight * q.q22
    var r0 = target.x - weight * q.q03
    var r1 = target.y - weight * q.q13
    var r2 = target.z - weight * q.q23
    var determinant = (
        a00 * (a11 * a22 - a12 * a12)
        - a01 * (a01 * a22 - a12 * a02)
        + a02 * (a01 * a12 - a11 * a02)
    )
    if abs(determinant) < 1.0e-12:
        return target
    var inverse = 1.0 / determinant
    return Vec3(
        (
            (a11 * a22 - a12 * a12) * r0
            + (a02 * a12 - a01 * a22) * r1
            + (a01 * a12 - a02 * a11) * r2
        )
        * inverse,
        (
            (a02 * a12 - a01 * a22) * r0
            + (a00 * a22 - a02 * a02) * r1
            + (a01 * a02 - a00 * a12) * r2
        )
        * inverse,
        (
            (a01 * a12 - a02 * a11) * r0
            + (a01 * a02 - a00 * a12) * r1
            + (a00 * a11 - a01 * a01) * r2
        )
        * inverse,
    )


def surface_relax(
    mut mesh: TriangleMesh, options: RemeshOptions
) raises -> RemeshStats:
    """Improve triangle regularity while preserving topology and source shape."""
    validate_mesh(mesh)
    if options.iterations < 0:
        raise Error("iterations cannot be negative")
    if options.relaxation < 0.0 or options.relaxation > 1.0:
        raise Error("relaxation must be in the range [0, 1]")
    if options.qem_weight < 0.0:
        raise Error("qem_weight cannot be negative")

    var source = mesh.copy()
    var vertex_faces = _build_vertex_faces(mesh)
    var source_face_normals = List[Vec3]()
    var source_face_areas = List[Float32]()
    source_face_normals.reserve(mesh.triangle_count())
    source_face_areas.reserve(mesh.triangle_count())
    for face in range(mesh.triangle_count()):
        source_face_normals.append(_face_normal(mesh, face))
        source_face_areas.append(_face_area(mesh, face))

    var quadrics = List[_Quadric]()
    var areas = List[Float32]()
    quadrics.reserve(mesh.vertex_count())
    areas.reserve(mesh.vertex_count())
    var boundaries = _boundary_vertices(mesh)
    var features = List[Bool]()
    features.reserve(mesh.vertex_count())
    for vertex in range(mesh.vertex_count()):
        var q = _Quadric()
        var area_sum: Float32 = 0.0
        for position in range(
            vertex_faces.offsets[vertex],
            vertex_faces.offsets[vertex + 1],
        ):
            var face = vertex_faces.faces[position]
            var normal = source_face_normals[face]
            var area = source_face_areas[face]
            var base = face * 3
            q.add_plane(
                normal, -normal.dot(mesh.vertices[mesh.indices[base]]), area
            )
            area_sum += area
        quadrics.append(q)
        areas.append(area_sum)
        features.append(
            _is_feature_vertex(
                vertex_faces,
                source_face_normals,
                vertex,
                options.feature_angle_degrees,
            )
            if options.preserve_features
            else False
        )

    var stats = RemeshStats()
    stats.iterations_requested = options.iterations
    stats.initial = compute_quality(mesh)
    stats.final = stats.initial
    for vertex in range(mesh.vertex_count()):
        if options.preserve_boundaries and boundaries[vertex]:
            stats.locked_boundary_vertices += 1
        if options.preserve_features and features[vertex]:
            stats.locked_feature_vertices += 1

    for _ in range(options.iterations):
        var face_areas = List[Float32]()
        var face_normals = List[Vec3]()
        var face_centroids = List[Vec3]()
        face_areas.reserve(mesh.triangle_count())
        face_normals.reserve(mesh.triangle_count())
        face_centroids.reserve(mesh.triangle_count())
        for face in range(mesh.triangle_count()):
            face_areas.append(_face_area(mesh, face))
            face_normals.append(_face_normal(mesh, face))
            face_centroids.append(_face_centroid(mesh, face))

        var proposals = List[Vec3]()
        proposals.reserve(mesh.vertex_count())
        for vertex in range(mesh.vertex_count()):
            var position = mesh.vertices[vertex]
            if (
                (options.preserve_boundaries and boundaries[vertex])
                or (options.preserve_features and features[vertex])
            ):
                proposals.append(position)
                continue
            var centroid = Vec3.zero()
            var normal = Vec3.zero()
            var total_area: Float32 = 0.0
            for face_position in range(
                vertex_faces.offsets[vertex],
                vertex_faces.offsets[vertex + 1],
            ):
                var face = vertex_faces.faces[face_position]
                var area = face_areas[face]
                centroid = centroid + face_centroids[face] * area
                normal = normal + face_normals[face] * area
                total_area += area
            if total_area <= 1.0e-12:
                proposals.append(position)
                continue
            centroid = centroid * (1.0 / total_area)
            normal = normal.normalized()
            var raw = centroid - position
            var tangent = raw - normal * raw.dot(normal)
            var target = position + tangent * options.relaxation
            var regularized = _solve_qem_target(
                target, quadrics[vertex], areas[vertex], options.qem_weight
            )
            proposals.append(
                _closest_point_on_source(
                    regularized, vertex, source, vertex_faces
                )
            )

        var accepted = False
        var accepted_quality = stats.final
        var accepted_vertices = List[Vec3]()
        accepted_vertices.reserve(mesh.vertex_count())
        for step in [
            Float32(1.0), Float32(0.5), Float32(0.25), Float32(0.125)
        ]:
            var candidate = mesh.copy()
            for vertex in range(candidate.vertex_count()):
                var blended = (
                    mesh.vertices[vertex]
                    + (proposals[vertex] - mesh.vertices[vertex]) * step
                )
                candidate.vertices[vertex] = _closest_point_on_source(
                    blended, vertex, source, vertex_faces
                )
            var quality = compute_quality(candidate)
            if quality.objective <= stats.final.objective + 1.0e-7:
                accepted = True
                accepted_quality = quality
                for point in candidate.vertices:
                    accepted_vertices.append(point)
                break
        if not accepted:
            stats.converged = True
            break
        var improvement = stats.final.objective - accepted_quality.objective
        for vertex in range(mesh.vertex_count()):
            mesh.vertices[vertex] = accepted_vertices[vertex]
        stats.final = accepted_quality
        stats.iterations_completed += 1
        if improvement <= options.convergence:
            stats.converged = True
            break

    for vertex in range(mesh.vertex_count()):
        var projected = _closest_point_on_source(
            mesh.vertices[vertex], vertex, source, vertex_faces
        )
        var deviation = (mesh.vertices[vertex] - projected).length()
        if deviation > stats.max_surface_deviation:
            stats.max_surface_deviation = deviation
    return stats
