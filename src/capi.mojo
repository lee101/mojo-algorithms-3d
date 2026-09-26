"""Zero-copy C ABI for high-throughput triangle-soup metrics."""

from std.math import sqrt
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float32]()
comptime QUALITY_PARALLEL_THRESHOLD = 4_000_000
comptime MAX_QUALITY_WORKERS = 16


def fp(address: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=address)


@always_inline
def quality_chunk(
    coordinates: F32Ptr,
    triangle_count: Int,
    begin: Int,
    end: Int,
    partials: F32Ptr,
    row: Int,
):
    var edge_accumulator = SIMD[DType.float32, W](0.0)
    var edge_square_accumulator = SIMD[DType.float32, W](0.0)
    var quality_accumulator = SIMD[DType.float32, W](0.0)
    var quality_minimum = SIMD[DType.float32, W](1.0)
    var triangle = begin
    while triangle + W <= end:
        var ax = coordinates.load[width=W](triangle)
        var ay = coordinates.load[width=W](triangle_count + triangle)
        var az = coordinates.load[width=W](2 * triangle_count + triangle)
        var bx = coordinates.load[width=W](3 * triangle_count + triangle)
        var by = coordinates.load[width=W](4 * triangle_count + triangle)
        var bz = coordinates.load[width=W](5 * triangle_count + triangle)
        var cx = coordinates.load[width=W](6 * triangle_count + triangle)
        var cy = coordinates.load[width=W](7 * triangle_count + triangle)
        var cz = coordinates.load[width=W](8 * triangle_count + triangle)

        var abx = ax - bx
        var aby = ay - by
        var abz = az - bz
        var bcx = bx - cx
        var bcy = by - cy
        var bcz = bz - cz
        var cax = cx - ax
        var cay = cy - ay
        var caz = cz - az
        var ab = sqrt(abx * abx + aby * aby + abz * abz)
        var bc = sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
        var ca = sqrt(cax * cax + cay * cay + caz * caz)
        edge_accumulator += ab + bc + ca
        edge_square_accumulator += ab * ab + bc * bc + ca * ca

        var acx = cx - ax
        var acy = cy - ay
        var acz = cz - az
        var cross_x = aby * acz - abz * acy
        var cross_y = abz * acx - abx * acz
        var cross_z = abx * acy - aby * acx
        var twice_area = sqrt(
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
        )
        var denominator = ab * ab + bc * bc + ca * ca
        var quality = denominator.gt(1.0e-12).select(
            2.0 * sqrt(Float32(3.0)) * twice_area / denominator,
            SIMD[DType.float32, W](0.0),
        )
        quality_accumulator += quality
        quality_minimum = min(quality_minimum, quality)
        triangle += W

    var edge_sum = edge_accumulator.reduce_add()
    var edge_square_sum = edge_square_accumulator.reduce_add()
    var quality_sum = quality_accumulator.reduce_add()
    var min_quality: Float32 = 1.0
    for lane in range(W):
        min_quality = min(min_quality, quality_minimum[lane])

    while triangle < end:
        var ax = coordinates[triangle]
        var ay = coordinates[triangle_count + triangle]
        var az = coordinates[2 * triangle_count + triangle]
        var bx = coordinates[3 * triangle_count + triangle]
        var by = coordinates[4 * triangle_count + triangle]
        var bz = coordinates[5 * triangle_count + triangle]
        var cx = coordinates[6 * triangle_count + triangle]
        var cy = coordinates[7 * triangle_count + triangle]
        var cz = coordinates[8 * triangle_count + triangle]

        var abx = ax - bx
        var aby = ay - by
        var abz = az - bz
        var bcx = bx - cx
        var bcy = by - cy
        var bcz = bz - cz
        var cax = cx - ax
        var cay = cy - ay
        var caz = cz - az
        var ab = sqrt(abx * abx + aby * aby + abz * abz)
        var bc = sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
        var ca = sqrt(cax * cax + cay * cay + caz * caz)
        edge_sum += ab + bc + ca
        edge_square_sum += ab * ab + bc * bc + ca * ca

        var acx = cx - ax
        var acy = cy - ay
        var acz = cz - az
        var cross_x = aby * acz - abz * acy
        var cross_y = abz * acx - abx * acz
        var cross_z = abx * acy - aby * acx
        var twice_area = sqrt(
            cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
        )
        var denominator = ab * ab + bc * bc + ca * ca
        var quality = (
            2.0 * sqrt(Float32(3.0)) * twice_area / denominator
            if denominator > 1.0e-12
            else 0.0
        )
        quality_sum += quality
        min_quality = min(min_quality, quality)
        triangle += 1

    var partial = row * 4
    partials[partial] = edge_sum
    partials[partial + 1] = edge_square_sum
    partials[partial + 2] = quality_sum
    partials[partial + 3] = min_quality


@export("a3d_triangle_soup_quality")
def a3d_triangle_soup_quality(
    coordinates_address: Int,
    triangle_count: Int,
    result_address: Int,
    scratch_address: Int,
    scratch_rows: Int,
) abi("C"):
    var coordinates = fp(coordinates_address)
    var result = fp(result_address)
    var partials = fp(scratch_address)
    var tasks = 1

    if (
        triangle_count >= QUALITY_PARALLEL_THRESHOLD
        and scratch_rows > 1
    ):
        tasks = min(
            scratch_rows,
            min(MAX_QUALITY_WORKERS, num_physical_cores()),
        )
        for task in range(tasks):
            var begin = triangle_count * task // tasks
            var end = triangle_count * (task + 1) // tasks
            quality_chunk(
                coordinates, triangle_count, begin, end, partials, task
            )
    else:
        quality_chunk(
            coordinates, triangle_count, 0, triangle_count, partials, 0
        )

    var edge_sum: Float32 = 0.0
    var edge_square_sum: Float32 = 0.0
    var quality_sum: Float32 = 0.0
    var min_quality: Float32 = 1.0
    for task in range(tasks):
        var partial = task * 4
        edge_sum += partials[partial]
        edge_square_sum += partials[partial + 1]
        quality_sum += partials[partial + 2]
        min_quality = min(min_quality, partials[partial + 3])

    if triangle_count <= 0:
        result[0] = 0.0
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 1.0
        result[4] = 0.0
        result[5] = Float32(tasks)
        return

    var edge_count = Float32(3 * triangle_count)
    var mean_edge_length = edge_sum / edge_count
    var variance = max(
        Float32(0.0),
        edge_square_sum / edge_count - mean_edge_length * mean_edge_length,
    )
    var coefficient_of_variation = (
        sqrt(variance) / mean_edge_length
        if mean_edge_length > 1.0e-12
        else 0.0
    )
    var mean_quality = quality_sum / Float32(triangle_count)
    result[0] = mean_edge_length
    result[1] = coefficient_of_variation
    result[2] = mean_quality
    result[3] = min_quality
    result[4] = coefficient_of_variation + 0.35 * (1.0 - mean_quality)
    result[5] = Float32(tasks)


@export("a3d_simd_width")
def a3d_simd_width() abi("C") -> Int:
    return W


@export("a3d_quality_parallel_threshold")
def a3d_quality_parallel_threshold() abi("C") -> Int:
    return QUALITY_PARALLEL_THRESHOLD
