"""C ABI for path flattening and antialiased SVG rasterization."""

from std.algorithm import parallelize
from std.gpu.host import DeviceContext
from std.math import ceil, floor, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def transformed_x(x: Float64, y: Float64, matrix: FPtr) -> Float64:
    return matrix[0] * x + matrix[2] * y + matrix[4]


def transformed_y(x: Float64, y: Float64, matrix: FPtr) -> Float64:
    return matrix[1] * x + matrix[3] * y + matrix[5]


def append_edge(
    edges: FPtr,
    edge_stride: Int,
    count: Int,
    x0: Float64,
    y0: Float64,
    x1: Float64,
    y1: Float64,
    matrix: FPtr,
):
    var tx0 = transformed_x(x0, y0, matrix)
    var ty0 = transformed_y(x0, y0, matrix)
    var tx1 = transformed_x(x1, y1, matrix)
    var ty1 = transformed_y(x1, y1, matrix)
    var dx = tx1 - tx0
    var dy = ty1 - ty0
    var denom = dx * dx + dy * dy
    edges[count] = tx0
    edges[edge_stride + count] = ty0
    edges[edge_stride * 2 + count] = tx1
    edges[edge_stride * 3 + count] = ty1
    edges[edge_stride * 4 + count] = dx
    edges[edge_stride * 5 + count] = dy
    edges[edge_stride * 6 + count] = 1.0 / denom if denom != 0.0 else 0.0
    edges[edge_stride * 7 + count] = dx / dy if dy != 0.0 else 0.0


@export("mcs_flatten")
def mcs_flatten(
    commands_addr: Int,
    command_count: Int,
    matrix_addr: Int,
    edges_addr: Int,
    edge_stride: Int,
) abi("C") -> Int:
    var commands = FPtr(unsafe_from_address=commands_addr)
    var matrix = FPtr(unsafe_from_address=matrix_addr)
    var edges = FPtr(unsafe_from_address=edges_addr)
    var count = 0
    var cx = 0.0
    var cy = 0.0
    var sx = 0.0
    var sy = 0.0
    var have_subpath = False
    for i in range(command_count):
        var base = i * 8
        var op = Int(commands[base])
        if op == 0:
            cx = commands[base + 1]
            cy = commands[base + 2]
            sx = cx
            sy = cy
            have_subpath = True
        elif op == 1:
            var nx = commands[base + 1]
            var ny = commands[base + 2]
            append_edge(edges, edge_stride, count, cx, cy, nx, ny, matrix)
            count += 1
            cx = nx
            cy = ny
        elif op == 2:
            var x0 = cx
            var y0 = cy
            var c1x = commands[base + 1]
            var c1y = commands[base + 2]
            var c2x = commands[base + 3]
            var c2y = commands[base + 4]
            var ex = commands[base + 5]
            var ey = commands[base + 6]
            var steps = max(2, Int(commands[base + 7]))
            var px = x0
            var py = y0
            for j in range(1, steps + 1):
                var t = Float64(j) / Float64(steps)
                var u = 1.0 - t
                var nx = u * u * u * x0 + 3.0 * u * u * t * c1x + 3.0 * u * t * t * c2x + t * t * t * ex
                var ny = u * u * u * y0 + 3.0 * u * u * t * c1y + 3.0 * u * t * t * c2y + t * t * t * ey
                append_edge(edges, edge_stride, count, px, py, nx, ny, matrix)
                count += 1
                px = nx
                py = ny
            cx = ex
            cy = ey
        elif op == 3:
            var x0 = cx
            var y0 = cy
            var qx = commands[base + 1]
            var qy = commands[base + 2]
            var ex = commands[base + 3]
            var ey = commands[base + 4]
            var steps = max(2, Int(commands[base + 7]))
            var px = x0
            var py = y0
            for j in range(1, steps + 1):
                var t = Float64(j) / Float64(steps)
                var u = 1.0 - t
                var nx = u * u * x0 + 2.0 * u * t * qx + t * t * ex
                var ny = u * u * y0 + 2.0 * u * t * qy + t * t * ey
                append_edge(edges, edge_stride, count, px, py, nx, ny, matrix)
                count += 1
                px = nx
                py = ny
            cx = ex
            cy = ey
        elif op == 4 and have_subpath:
            if cx != sx or cy != sy:
                append_edge(edges, edge_stride, count, cx, cy, sx, sy, matrix)
                count += 1
            cx = sx
            cy = sy
    return count


def fill_contains(
    edges: FPtr,
    edge_count: Int,
    edge_stride: Int,
    px: Float64,
    py: Float64,
    rule: Int,
) -> Bool:
    comptime W = simdwidthof[DType.float64]()
    var winding = 0
    var i = 0
    var vector_winding = SIMD[DType.int64, W](0)
    while edge_count >= W * 2 and i + W <= edge_count:
        var x0 = edges.load[width=W](i)
        var y0 = edges.load[width=W](edge_stride + i)
        var y1 = edges.load[width=W](edge_stride * 3 + i)
        var crosses = (y0.le(py) & y1.gt(py)) | (y1.le(py) & y0.gt(py))
        if not any(crosses):
            i += W
            continue
        var slope = edges.load[width=W](edge_stride * 7 + i)
        var cross_x = x0 + (py - y0) * slope
        var right = crosses & cross_x.gt(px)
        if rule == 1:
            vector_winding += right.cast[DType.int64]()
        else:
            var direction = y1.gt(y0).select(
                SIMD[DType.int64, W](1), SIMD[DType.int64, W](-1)
            )
            vector_winding += right.select(direction, SIMD[DType.int64, W](0))
        i += W
    winding = Int(vector_winding.reduce_add())
    while i < edge_count:
        var x0 = edges[i]
        var y0 = edges[edge_stride + i]
        var y1 = edges[edge_stride * 3 + i]
        if (y0 <= py and y1 > py) or (y1 <= py and y0 > py):
            var cross_x = x0 + (py - y0) * edges[edge_stride * 7 + i]
            if cross_x > px:
                if rule == 1:
                    winding = 1 - winding
                elif y1 > y0:
                    winding += 1
                else:
                    winding -= 1
        i += 1
    return winding % 2 != 0 if rule == 1 else winding != 0


def edge_distance2(
    edges: FPtr,
    edge_count: Int,
    edge_stride: Int,
    px: Float64,
    py: Float64,
    round_caps: Bool,
    max_distance: Float64,
) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    var best = 1.7976931348623157e308
    var i = 0
    while edge_count >= W * 2 and i + W <= edge_count:
        var x0 = edges.load[width=W](i)
        var y0 = edges.load[width=W](edge_stride + i)
        var x1 = edges.load[width=W](edge_stride * 2 + i)
        var y1 = edges.load[width=W](edge_stride * 3 + i)
        var valid = (
            max(x0, x1).ge(px - max_distance)
            & min(x0, x1).le(px + max_distance)
            & max(y0, y1).ge(py - max_distance)
            & min(y0, y1).le(py + max_distance)
        )
        if not any(valid):
            i += W
            continue
        var dx = edges.load[width=W](edge_stride * 4 + i)
        var dy = edges.load[width=W](edge_stride * 5 + i)
        var inv_denom = edges.load[width=W](edge_stride * 6 + i)
        valid &= inv_denom.ne(0.0)
        var t = ((px - x0) * dx + (py - y0) * dy) * inv_denom
        if round_caps:
            t = max(
                SIMD[DType.float64, W](0.0),
                min(SIMD[DType.float64, W](1.0), t),
            )
        else:
            valid &= t.ge(0.0) & t.le(1.0)
        var ex = px - (x0 + t * dx)
        var ey = py - (y0 + t * dy)
        var distance2 = ex * ex + ey * ey
        var candidates = valid.select(
            distance2, SIMD[DType.float64, W](1.7976931348623157e308)
        )
        best = min(best, Float64(candidates.reduce_min()))
        i += W
    while i < edge_count:
        var x0 = edges[i]
        var y0 = edges[edge_stride + i]
        var x1 = edges[edge_stride * 2 + i]
        var y1 = edges[edge_stride * 3 + i]
        if (
            max(x0, x1) < px - max_distance
            or min(x0, x1) > px + max_distance
            or max(y0, y1) < py - max_distance
            or min(y0, y1) > py + max_distance
        ):
            i += 1
            continue
        var dx = edges[edge_stride * 4 + i]
        var dy = edges[edge_stride * 5 + i]
        var inv_denom = edges[edge_stride * 6 + i]
        if inv_denom == 0.0:
            i += 1
            continue
        var t = ((px - x0) * dx + (py - y0) * dy) * inv_denom
        if round_caps:
            t = max(0.0, min(1.0, t))
        elif t < 0.0 or t > 1.0:
            i += 1
            continue
        var qx = x0 + t * dx
        var qy = y0 + t * dy
        var ex = px - qx
        var ey = py - qy
        best = min(best, ex * ex + ey * ey)
        i += 1
    return best


def byte_value(value: Float64) -> UInt8:
    return UInt8(max(0, min(255, Int(floor(value * 255.0 + 0.5)))))


@always_inline
def paint_pixel(
    canvas: BPtr,
    width: Int,
    x: Int,
    y: Int,
    coverage: Float64,
    paint_kind: Int,
    paint: FPtr,
    stops: FPtr,
    stop_count: Int,
    opacity: Float64,
):
    var red = paint[0]
    var green = paint[1]
    var blue = paint[2]
    var alpha = paint[3]
    if paint_kind != 0 and stop_count > 0:
        var px = Float64(x) + 0.5
        var py = Float64(y) + 0.5
        var t = 0.0
        if paint_kind == 1:
            t = paint[4] * px + paint[5] * py + paint[6]
        else:
            var dx = px - paint[4]
            var dy = py - paint[5]
            if paint[6] > 0.0:
                t = sqrt(dx * dx + dy * dy) / paint[6]
        t = max(0.0, min(1.0, t))
        var upper = 0
        while upper < stop_count and stops[upper * 5] < t:
            upper += 1
        if upper == 0:
            red = stops[1]
            green = stops[2]
            blue = stops[3]
            alpha = stops[4]
        elif upper >= stop_count:
            var lower = (stop_count - 1) * 5
            red = stops[lower + 1]
            green = stops[lower + 2]
            blue = stops[lower + 3]
            alpha = stops[lower + 4]
        else:
            var lo = (upper - 1) * 5
            var hi = upper * 5
            var span = stops[hi] - stops[lo]
            var f = (t - stops[lo]) / span if span > 0.0 else 0.0
            red = stops[lo + 1] * (1.0 - f) + stops[hi + 1] * f
            green = stops[lo + 2] * (1.0 - f) + stops[hi + 2] * f
            blue = stops[lo + 3] * (1.0 - f) + stops[hi + 3] * f
            alpha = stops[lo + 4] * (1.0 - f) + stops[hi + 4] * f
    var src_alpha = max(0.0, min(1.0, alpha * opacity * coverage))
    var index = (y * width + x) * 4
    if src_alpha >= 1.0:
        canvas[index] = byte_value(red)
        canvas[index + 1] = byte_value(green)
        canvas[index + 2] = byte_value(blue)
        canvas[index + 3] = 255
        return
    var dst_alpha = Float64(canvas[index + 3]) / 255.0
    var result_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    if result_alpha <= 0.0:
        return
    var keep = dst_alpha * (1.0 - src_alpha)
    var dst_red = Float64(canvas[index]) / 255.0
    var dst_green = Float64(canvas[index + 1]) / 255.0
    var dst_blue = Float64(canvas[index + 2]) / 255.0
    canvas[index] = byte_value((red * src_alpha + dst_red * keep) / result_alpha)
    canvas[index + 1] = byte_value((green * src_alpha + dst_green * keep) / result_alpha)
    canvas[index + 2] = byte_value((blue * src_alpha + dst_blue * keep) / result_alpha)
    canvas[index + 3] = byte_value(result_alpha)


def draw_rect_row(
    canvas: BPtr,
    width: Int,
    left: Int,
    right: Int,
    y: Int,
    rect_left: Float64,
    rect_top: Float64,
    rect_right: Float64,
    rect_bottom: Float64,
    paint_kind: Int,
    paint: FPtr,
    stops: FPtr,
    stop_count: Int,
    opacity: Float64,
    sample_count: Int,
    inv_samples: Float64,
):
    var y_hits = 0
    for sy in range(sample_count):
        var py = Float64(y) + (Float64(sy) + 0.5) / Float64(sample_count)
        if py >= rect_top and py < rect_bottom:
            y_hits += 1
    if y_hits == 0:
        return
    for x in range(left, right):
        var x_hits = 0
        for sx in range(sample_count):
            var px = Float64(x) + (Float64(sx) + 0.5) / Float64(sample_count)
            if px >= rect_left and px < rect_right:
                x_hits += 1
        if x_hits == 0:
            continue
        paint_pixel(
            canvas, width, x, y, Float64(x_hits * y_hits) * inv_samples,
            paint_kind, paint, stops, stop_count, opacity,
        )


def draw_row(
    canvas: BPtr,
    width: Int,
    edges: FPtr,
    edge_count: Int,
    edge_stride: Int,
    left: Int,
    right: Int,
    y: Int,
    draw_mode: Int,
    fill_rule: Int,
    half_width: Float64,
    half_width2: Float64,
    round_caps: Int,
    paint_kind: Int,
    paint: FPtr,
    stops: FPtr,
    stop_count: Int,
    opacity: Float64,
    sample_count: Int,
    inv_samples: Float64,
):
    for x in range(left, right):
        var hits = 0
        var center_x = Float64(x) + 0.5
        var center_y = Float64(y) + 0.5
        var distance_limit = 1.0 if draw_mode == 0 else half_width + 1.0
        var center_distance2 = edge_distance2(
            edges, edge_count, edge_stride, center_x, center_y,
            draw_mode == 0 or round_caps != 0, distance_limit,
        )
        var needs_samples = center_distance2 <= 1.0
        if draw_mode != 0:
            var center_distance = sqrt(center_distance2)
            needs_samples = abs(center_distance - half_width) <= 1.0
            if not needs_samples and center_distance < half_width:
                hits = sample_count * sample_count
        elif not needs_samples and fill_contains(
            edges, edge_count, edge_stride, center_x, center_y, fill_rule
        ):
            hits = sample_count * sample_count
        if needs_samples:
            for sy in range(sample_count):
                var py = Float64(y) + (Float64(sy) + 0.5) / Float64(sample_count)
                for sx in range(sample_count):
                    var px = Float64(x) + (Float64(sx) + 0.5) / Float64(sample_count)
                    if draw_mode == 0:
                        if fill_contains(
                            edges, edge_count, edge_stride, px, py, fill_rule
                        ):
                            hits += 1
                    elif edge_distance2(
                        edges, edge_count, edge_stride, px, py, round_caps != 0,
                        half_width,
                    ) <= half_width2:
                        hits += 1
        if hits == 0:
            continue
        paint_pixel(
            canvas, width, x, y, Float64(hits) * inv_samples,
            paint_kind, paint, stops, stop_count, opacity,
        )


@export("mcs_draw_path")
def mcs_draw_path(
    canvas_addr: Int,
    width: Int,
    height: Int,
    edges_addr: Int,
    edge_count: Int,
    edge_stride: Int,
    x_min: Int,
    y_min: Int,
    x_max: Int,
    y_max: Int,
    draw_mode: Int,
    fill_rule: Int,
    stroke_width: Float64,
    round_caps: Int,
    paint_kind: Int,
    paint_addr: Int,
    stops_addr: Int,
    stop_count: Int,
    opacity: Float64,
    samples: Int,
) abi("C"):
    var canvas = BPtr(unsafe_from_address=canvas_addr)
    var edges = FPtr(unsafe_from_address=edges_addr)
    var paint = FPtr(unsafe_from_address=paint_addr)
    var stops = FPtr(unsafe_from_address=stops_addr)
    var sample_count = max(1, samples)
    var inv_samples = 1.0 / Float64(sample_count * sample_count)
    var half_width2 = stroke_width * stroke_width * 0.25
    var half_width = stroke_width * 0.5
    var left = max(0, x_min)
    var top = max(0, y_min)
    var right = min(width, x_max)
    var bottom = min(height, y_max)
    var rows = bottom - top
    var pixel_count = rows * (right - left)
    var axis_rect = draw_mode == 0 and edge_count == 4
    var rect_left = 1.7976931348623157e308
    var rect_top = 1.7976931348623157e308
    var rect_right = -1.7976931348623157e308
    var rect_bottom = -1.7976931348623157e308
    if axis_rect:
        for i in range(4):
            var x0 = edges[i]
            var y0 = edges[edge_stride + i]
            var x1 = edges[edge_stride * 2 + i]
            var y1 = edges[edge_stride * 3 + i]
            if x0 != x1 and y0 != y1:
                axis_rect = False
            rect_left = min(rect_left, min(x0, x1))
            rect_top = min(rect_top, min(y0, y1))
            rect_right = max(rect_right, max(x0, x1))
            rect_bottom = max(rect_bottom, max(y0, y1))
    if axis_rect:
        if pixel_count >= 65536 and rows >= 8:
            @parameter
            def rect_work(row: Int):
                draw_rect_row(
                    canvas, width, left, right, top + row, rect_left, rect_top,
                    rect_right, rect_bottom, paint_kind, paint, stops, stop_count,
                    opacity, sample_count, inv_samples,
                )

            try:
                var cpu_ctx = DeviceContext(api="cpu")
                parallelize[rect_work](rows, ctx=cpu_ctx)
            except:
                for y in range(top, bottom):
                    draw_rect_row(
                        canvas, width, left, right, y, rect_left, rect_top,
                        rect_right, rect_bottom, paint_kind, paint, stops,
                        stop_count, opacity, sample_count, inv_samples,
                    )
        else:
            for y in range(top, bottom):
                draw_rect_row(
                    canvas, width, left, right, y, rect_left, rect_top,
                    rect_right, rect_bottom, paint_kind, paint, stops, stop_count,
                    opacity, sample_count, inv_samples,
                )
        return
    if pixel_count >= 65536 and rows >= 8:
        @parameter
        def work(row: Int):
            draw_row(
                canvas, width, edges, edge_count, edge_stride, left, right, top + row,
                draw_mode, fill_rule, half_width, half_width2, round_caps,
                paint_kind, paint, stops, stop_count, opacity, sample_count,
                inv_samples,
            )

        try:
            var cpu_ctx = DeviceContext(api="cpu")
            parallelize[work](rows, ctx=cpu_ctx)
        except:
            for y in range(top, bottom):
                draw_row(
                    canvas, width, edges, edge_count, edge_stride, left, right, y,
                    draw_mode, fill_rule, half_width, half_width2, round_caps,
                    paint_kind, paint, stops, stop_count, opacity, sample_count,
                    inv_samples,
                )
    else:
        for y in range(top, bottom):
            draw_row(
                canvas, width, edges, edge_count, edge_stride, left, right, y,
                draw_mode, fill_rule, half_width, half_width2, round_caps,
                paint_kind, paint, stops, stop_count, opacity, sample_count,
                inv_samples,
            )


@export("mcs_draw_rects")
def mcs_draw_rects(
    canvas_addr: Int,
    width: Int,
    height: Int,
    rects_addr: Int,
    rect_count: Int,
    samples: Int,
) abi("C"):
    var canvas = BPtr(unsafe_from_address=canvas_addr)
    var rects = FPtr(unsafe_from_address=rects_addr)
    var sample_count = max(1, samples)
    var inv_samples = 1.0 / Float64(sample_count * sample_count)
    for i in range(rect_count):
        var base = i * 8
        var rect_left = min(rects[base], rects[base + 2])
        var rect_top = min(rects[base + 1], rects[base + 3])
        var rect_right = max(rects[base], rects[base + 2])
        var rect_bottom = max(rects[base + 1], rects[base + 3])
        var left = max(0, Int(floor(rect_left - 1.0)))
        var top = max(0, Int(floor(rect_top - 1.0)))
        var right = min(width, Int(ceil(rect_right + 1.0)))
        var bottom = min(height, Int(ceil(rect_bottom + 1.0)))
        for y in range(top, bottom):
            draw_rect_row(
                canvas, width, left, right, y, rect_left, rect_top, rect_right,
                rect_bottom, 0, rects + base + 4, rects + base + 4, 0, 1.0,
                sample_count, inv_samples,
            )


@export("mcs_premultiply")
def mcs_premultiply(src_addr: Int, dst_addr: Int, pixel_count: Int) abi("C"):
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    for i in range(pixel_count):
        var base = i * 4
        var alpha = Float64(src[base + 3]) / 255.0
        dst[base] = UInt8(floor(Float64(src[base]) * alpha + 0.5))
        dst[base + 1] = UInt8(floor(Float64(src[base + 1]) * alpha + 0.5))
        dst[base + 2] = UInt8(floor(Float64(src[base + 2]) * alpha + 0.5))
        dst[base + 3] = src[base + 3]
