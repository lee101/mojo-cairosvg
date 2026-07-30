"""SVG document preparation for the Mojo scan converter."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
import os
import re
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageColor

from ._lib import draw_path, draw_rects, flatten

_NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_PATH_TOKEN = re.compile(rf"[AaCcHhLlMmQqSsTtVvZz]|{_NUMBER}")
_TRANSFORM = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
_LENGTH = re.compile(rf"^\s*({_NUMBER})\s*(%|px|pt|pc|mm|cm|in)?\s*$")
_URL_PAINT = re.compile(r"^url\(\s*#([^) ]+)\s*\)$")

_INHERITED = {
    "color", "fill", "fill-opacity", "fill-rule", "stroke", "stroke-opacity",
    "stroke-width", "stroke-linecap", "stroke-linejoin", "visibility",
}
_DEFAULT_STYLE = {
    "color": "black",
    "fill": "black",
    "fill-opacity": "1",
    "fill-rule": "nonzero",
    "stroke": "none",
    "stroke-opacity": "1",
    "stroke-width": "1",
    "stroke-linecap": "butt",
    "stroke-linejoin": "miter",
    "opacity": "1",
    "display": "inline",
    "visibility": "visible",
}


@dataclass
class Gradient:
    kind: int
    attrs: dict[str, str]
    stops: np.ndarray


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _style(element: ET.Element, inherited: dict[str, str]) -> dict[str, str]:
    result = {key: value for key, value in inherited.items() if key in _INHERITED}
    for key in set(_DEFAULT_STYLE) | set(element.attrib):
        if key in element.attrib:
            result[key] = element.attrib[key]
    for declaration in element.attrib.get("style", "").split(";"):
        if ":" in declaration:
            key, value = declaration.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def _length(
    value: str | None,
    reference: float,
    dpi: float,
    default: float = 0.0,
) -> float:
    if value is None:
        return default
    match = _LENGTH.match(value)
    if not match:
        raise ValueError(f"unsupported SVG length: {value!r}")
    number = float(match.group(1))
    unit = match.group(2) or "px"
    if unit == "%":
        return number * reference / 100.0
    factors = {
        "px": 1.0,
        "in": dpi,
        "cm": dpi / 2.54,
        "mm": dpi / 25.4,
        "pt": dpi / 72.0,
        "pc": dpi / 6.0,
    }
    return number * factors[unit]


def _fraction(value: str | None, default: float) -> float:
    if value is None:
        return default
    return float(value[:-1]) / 100.0 if value.strip().endswith("%") else float(value)


def _matrix_values(matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1], matrix[0, 2], matrix[1, 2]],
        dtype=np.float64,
    )


def _parse_transform(value: str | None) -> np.ndarray:
    result = np.eye(3)
    if not value:
        return result
    for name, args_text in _TRANSFORM.findall(value):
        values = [float(item) for item in re.findall(_NUMBER, args_text)]
        transform = np.eye(3)
        name = name.lower()
        if name == "matrix" and len(values) == 6:
            a, b, c, d, e, f = values
            transform = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        elif name == "translate" and values:
            transform[0, 2] = values[0]
            transform[1, 2] = values[1] if len(values) > 1 else 0.0
        elif name == "scale" and values:
            transform[0, 0] = values[0]
            transform[1, 1] = values[1] if len(values) > 1 else values[0]
        elif name == "rotate" and values:
            angle = math.radians(values[0])
            rotation = np.array([
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ])
            if len(values) >= 3:
                cx, cy = values[1:3]
                transform = _parse_transform(f"translate({cx} {cy})") @ rotation @ \
                    _parse_transform(f"translate({-cx} {-cy})")
            else:
                transform = rotation
        elif name in {"skewx", "skewy"} and values:
            axis = 1 if name == "skewx" else 0
            transform[1 - axis, axis] = math.tan(math.radians(values[0]))
        else:
            raise ValueError(f"unsupported transform: {name}({args_text})")
        result = result @ transform
    return result


def _command(op: int, *values: float, steps: int = 1) -> list[float]:
    row = [float(op), *map(float, values)]
    row.extend([0.0] * (8 - len(row)))
    row[7] = float(steps)
    return row


def _curve_steps(points: tuple[float, ...]) -> int:
    xs = points[0::2]
    ys = points[1::2]
    length = sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))
    return max(6, min(48, math.ceil(length / 5.0)))


def _arc_cubics(
    x0: float,
    y0: float,
    rx: float,
    ry: float,
    rotation: float,
    large: bool,
    sweep: bool,
    x1: float,
    y1: float,
) -> list[tuple[float, ...]]:
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0 or (x0 == x1 and y0 == y1):
        return []
    phi = math.radians(rotation % 360)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx, dy = (x0 - x1) / 2, (y0 - y1) / 2
    xp = cos_phi * dx + sin_phi * dy
    yp = -sin_phi * dx + cos_phi * dy
    radius_scale = xp * xp / (rx * rx) + yp * yp / (ry * ry)
    if radius_scale > 1:
        factor = math.sqrt(radius_scale)
        rx *= factor
        ry *= factor
    sign = -1 if large == sweep else 1
    numerator = max(0.0, rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp)
    denominator = rx * rx * yp * yp + ry * ry * xp * xp
    coefficient = sign * math.sqrt(numerator / denominator) if denominator else 0.0
    cxp = coefficient * rx * yp / ry
    cyp = -coefficient * ry * xp / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x0 + x1) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y0 + y1) / 2

    def angle(ux: float, uy: float, vx: float, vy: float) -> float:
        dot = ux * vx + uy * vy
        det = ux * vy - uy * vx
        return math.atan2(det, dot)

    ux, uy = (xp - cxp) / rx, (yp - cyp) / ry
    vx, vy = (-xp - cxp) / rx, (-yp - cyp) / ry
    start = angle(1, 0, ux, uy)
    delta = angle(ux, uy, vx, vy)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    if sweep and delta < 0:
        delta += 2 * math.pi
    pieces = max(1, math.ceil(abs(delta) / (math.pi / 2)))
    step = delta / pieces
    result = []
    for index in range(pieces):
        a0 = start + index * step
        a1 = a0 + step
        alpha = 4 / 3 * math.tan((a1 - a0) / 4)

        def point(angle_value: float) -> tuple[float, float]:
            return (
                cx + rx * cos_phi * math.cos(angle_value) - ry * sin_phi * math.sin(angle_value),
                cy + rx * sin_phi * math.cos(angle_value) + ry * cos_phi * math.sin(angle_value),
            )

        p0, p3 = point(a0), point(a1)
        d0 = (-rx * cos_phi * math.sin(a0) - ry * sin_phi * math.cos(a0),
              -rx * sin_phi * math.sin(a0) + ry * cos_phi * math.cos(a0))
        d1 = (-rx * cos_phi * math.sin(a1) - ry * sin_phi * math.cos(a1),
              -rx * sin_phi * math.sin(a1) + ry * cos_phi * math.cos(a1))
        result.append((
            p0[0] + alpha * d0[0], p0[1] + alpha * d0[1],
            p3[0] - alpha * d1[0], p3[1] - alpha * d1[1],
            p3[0], p3[1],
        ))
    return result


def _parse_path(data: str) -> np.ndarray:
    tokens = _PATH_TOKEN.findall(data.replace(",", " "))
    rows: list[list[float]] = []
    index = 0
    operation = ""
    x = y = start_x = start_y = 0.0
    last_cubic: tuple[float, float] | None = None
    last_quad: tuple[float, float] | None = None
    counts = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}
    while index < len(tokens):
        if tokens[index].isalpha():
            operation = tokens[index]
            index += 1
            if operation.upper() == "Z":
                rows.append(_command(4))
                x, y = start_x, start_y
                last_cubic = last_quad = None
                continue
        if not operation:
            raise ValueError("path data must start with a command")
        upper = operation.upper()
        if upper == "Z":
            operation = ""
            continue
        count = counts[upper]
        if index + count > len(tokens) or tokens[index].isalpha():
            raise ValueError(f"incomplete SVG path command {operation}")
        values = list(map(float, tokens[index:index + count]))
        index += count
        relative = operation.islower()
        ox, oy = x, y
        if upper == "M":
            nx, ny = values
            if relative:
                nx, ny = nx + x, ny + y
            rows.append(_command(0 if not rows or operation.upper() == "M" else 1, nx, ny))
            x, y = start_x, start_y = nx, ny
            operation = "l" if relative else "L"
        elif upper == "L":
            nx, ny = values
            x, y = (nx + x, ny + y) if relative else (nx, ny)
            rows.append(_command(1, x, y))
        elif upper == "H":
            x = values[0] + x if relative else values[0]
            rows.append(_command(1, x, y))
        elif upper == "V":
            y = values[0] + y if relative else values[0]
            rows.append(_command(1, x, y))
        elif upper == "C":
            c1x, c1y, c2x, c2y, nx, ny = values
            if relative:
                c1x, c1y, c2x, c2y, nx, ny = (
                    c1x + x, c1y + y, c2x + x, c2y + y, nx + x, ny + y)
            rows.append(_command(2, c1x, c1y, c2x, c2y, nx, ny,
                                 steps=_curve_steps((x, y, c1x, c1y, c2x, c2y, nx, ny))))
            x, y = nx, ny
            last_cubic, last_quad = (c2x, c2y), None
        elif upper == "S":
            c2x, c2y, nx, ny = values
            if relative:
                c2x, c2y, nx, ny = c2x + x, c2y + y, nx + x, ny + y
            c1x, c1y = (2 * x - last_cubic[0], 2 * y - last_cubic[1]) \
                if last_cubic else (x, y)
            rows.append(_command(2, c1x, c1y, c2x, c2y, nx, ny,
                                 steps=_curve_steps((x, y, c1x, c1y, c2x, c2y, nx, ny))))
            x, y = nx, ny
            last_cubic, last_quad = (c2x, c2y), None
        elif upper == "Q":
            qx, qy, nx, ny = values
            if relative:
                qx, qy, nx, ny = qx + x, qy + y, nx + x, ny + y
            rows.append(_command(3, qx, qy, nx, ny,
                                 steps=_curve_steps((x, y, qx, qy, nx, ny))))
            x, y = nx, ny
            last_quad, last_cubic = (qx, qy), None
        elif upper == "T":
            nx, ny = values
            if relative:
                nx, ny = nx + x, ny + y
            qx, qy = (2 * x - last_quad[0], 2 * y - last_quad[1]) if last_quad else (x, y)
            rows.append(_command(3, qx, qy, nx, ny,
                                 steps=_curve_steps((x, y, qx, qy, nx, ny))))
            x, y = nx, ny
            last_quad, last_cubic = (qx, qy), None
        elif upper == "A":
            rx, ry, rotation, large, sweep, nx, ny = values
            if relative:
                nx, ny = nx + x, ny + y
            cubics = _arc_cubics(x, y, rx, ry, rotation, bool(large), bool(sweep), nx, ny)
            if not cubics:
                rows.append(_command(1, nx, ny))
            for cubic in cubics:
                rows.append(_command(2, *cubic, steps=12))
            x, y = nx, ny
            last_cubic = last_quad = None
        if upper not in {"C", "S", "Q", "T"}:
            last_cubic = last_quad = None
        if (x, y) == (ox, oy) and upper not in {"M", "H", "V"}:
            pass
    return np.asarray(rows, dtype=np.float64).reshape((-1, 8))


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    k = 0.5522847498307936
    rows = [
        _command(0, cx + rx, cy),
        _command(2, cx + rx, cy + k * ry, cx + k * rx, cy + ry, cx, cy + ry, steps=12),
        _command(2, cx - k * rx, cy + ry, cx - rx, cy + k * ry, cx - rx, cy, steps=12),
        _command(2, cx - rx, cy - k * ry, cx - k * rx, cy - ry, cx, cy - ry, steps=12),
        _command(2, cx + k * rx, cy - ry, cx + rx, cy - k * ry, cx + rx, cy, steps=12),
        _command(4),
    ]
    return np.asarray(rows, dtype=np.float64)


def _geometry(element: ET.Element, width: float, height: float, dpi: float) -> np.ndarray:
    name = _tag(element)
    get_x = lambda key, default=0: _length(element.get(key), width, dpi, default)
    get_y = lambda key, default=0: _length(element.get(key), height, dpi, default)
    if name == "path":
        return _parse_path(element.get("d", ""))
    if name == "circle":
        radius = _length(element.get("r"), min(width, height), dpi)
        return _ellipse(get_x("cx"), get_y("cy"), radius, radius)
    if name == "ellipse":
        return _ellipse(get_x("cx"), get_y("cy"), get_x("rx"), get_y("ry"))
    if name == "rect":
        x, y, w, h = get_x("x"), get_y("y"), get_x("width"), get_y("height")
        rx = min(get_x("rx", get_y("ry", 0)), w / 2)
        ry = min(get_y("ry", rx), h / 2)
        if rx > 0 and ry > 0:
            k = 0.5522847498307936
            rows = [
                _command(0, x + rx, y), _command(1, x + w - rx, y),
                _command(2, x + w - rx + k * rx, y, x + w, y + ry - k * ry,
                         x + w, y + ry, steps=6),
                _command(1, x + w, y + h - ry),
                _command(2, x + w, y + h - ry + k * ry, x + w - rx + k * rx, y + h,
                         x + w - rx, y + h, steps=6),
                _command(1, x + rx, y + h),
                _command(2, x + rx - k * rx, y + h, x, y + h - ry + k * ry,
                         x, y + h - ry, steps=6),
                _command(1, x, y + ry),
                _command(2, x, y + ry - k * ry, x + rx - k * rx, y, x + rx, y, steps=6),
                _command(4),
            ]
        else:
            rows = [
                _command(0, x, y), _command(1, x + w, y), _command(1, x + w, y + h),
                _command(1, x, y + h), _command(4),
            ]
        return np.asarray(rows, dtype=np.float64)
    if name == "line":
        return np.asarray([
            _command(0, get_x("x1"), get_y("y1")),
            _command(1, get_x("x2"), get_y("y2")),
        ], dtype=np.float64)
    if name in {"polyline", "polygon"}:
        values = list(map(float, re.findall(_NUMBER, element.get("points", ""))))
        if len(values) < 4 or len(values) % 2:
            return np.empty((0, 8), dtype=np.float64)
        rows = [_command(0, values[0], values[1])]
        rows.extend(_command(1, values[i], values[i + 1]) for i in range(2, len(values), 2))
        if name == "polygon":
            rows.append(_command(4))
        return np.asarray(rows, dtype=np.float64)
    return np.empty((0, 8), dtype=np.float64)


def _color(value: str, current: str, negate: bool) -> np.ndarray:
    if value == "currentColor":
        value = current
    rgba = np.array(ImageColor.getcolor(value.strip(), "RGBA"), dtype=np.float64) / 255.0
    if negate:
        rgba[:3] = 1.0 - rgba[:3]
    return rgba


def _definitions(root: ET.Element, negate: bool) -> dict[str, Gradient]:
    result: dict[str, Gradient] = {}
    for element in root.iter():
        name = _tag(element)
        if name not in {"linearGradient", "radialGradient"} or not element.get("id"):
            continue
        if element.get("gradientTransform"):
            raise NotImplementedError("gradientTransform is not supported")
        if element.get("spreadMethod", "pad") != "pad":
            raise NotImplementedError("gradient spread methods are not supported")
        if name == "radialGradient" and (
            element.get("fx") is not None or element.get("fy") is not None
        ):
            raise NotImplementedError("focal radial gradients are not supported")
        stops = []
        for stop in element:
            if _tag(stop) != "stop":
                continue
            style = _style(stop, {})
            color = _color(style.get("stop-color", stop.get("stop-color", "black")), "black", negate)
            color[3] *= float(style.get("stop-opacity", stop.get("stop-opacity", "1")))
            stops.append([_fraction(stop.get("offset"), 0.0), *color])
        stops.sort(key=lambda row: row[0])
        if not stops:
            stops = [[0, 0, 0, 0, 1], [1, 0, 0, 0, 1]]
        result[element.get("id", "")] = Gradient(
            1 if name == "linearGradient" else 2,
            dict(element.attrib),
            np.asarray(stops, dtype=np.float64),
        )
    return result


def _paint(
    value: str,
    current: str,
    gradients: dict[str, Gradient],
    bounds: tuple[float, float, float, float],
    matrix: np.ndarray,
    negate: bool,
) -> tuple[int, np.ndarray, np.ndarray] | None:
    if value.strip().lower() == "none":
        return None
    match = _URL_PAINT.match(value.strip())
    params = np.zeros(8, dtype=np.float64)
    if not match:
        params[:4] = _color(value, current, negate)
        return 0, params, np.empty((0, 5), dtype=np.float64)
    gradient = gradients.get(match.group(1))
    if gradient is None:
        raise ValueError(f"unknown SVG paint server #{match.group(1)}")
    x0, y0, x1, y1 = bounds
    attrs = gradient.attrs
    object_box = attrs.get("gradientUnits", "objectBoundingBox") != "userSpaceOnUse"
    if gradient.kind == 1:
        if object_box:
            gx1 = _fraction(attrs.get("x1"), 0.0)
            gy1 = _fraction(attrs.get("y1"), 0.0)
            gx2 = _fraction(attrs.get("x2"), 1.0)
            gy2 = _fraction(attrs.get("y2"), 0.0)
            dx, dy = gx2 - gx1, gy2 - gy1
            denominator = dx * dx + dy * dy
            a = dx / ((x1 - x0) * denominator) if x1 != x0 and denominator else 0.0
            b = dy / ((y1 - y0) * denominator) if y1 != y0 and denominator else 0.0
            c = -(a * x0 + b * y0) - (gx1 * dx + gy1 * dy) / denominator \
                if denominator else 0.0
        else:
            gx1 = _length(attrs.get("x1"), x1 - x0, 96)
            gy1 = _length(attrs.get("y1"), y1 - y0, 96)
            gx2 = _length(attrs.get("x2"), x1 - x0, 96, 1)
            gy2 = _length(attrs.get("y2"), y1 - y0, 96)
            dx, dy = gx2 - gx1, gy2 - gy1
            denominator = dx * dx + dy * dy
            inverse = np.linalg.inv(matrix)
            a = (dx * inverse[0, 0] + dy * inverse[1, 0]) / denominator
            b = (dx * inverse[0, 1] + dy * inverse[1, 1]) / denominator
            c = (dx * (inverse[0, 2] - gx1) + dy * (inverse[1, 2] - gy1)) / denominator
        params[4:7] = [a, b, c]
    else:
        if object_box:
            cx = x0 + _fraction(attrs.get("cx"), 0.5) * (x1 - x0)
            cy = y0 + _fraction(attrs.get("cy"), 0.5) * (y1 - y0)
            radius = _fraction(attrs.get("r"), 0.5) * max(x1 - x0, y1 - y0)
        else:
            point = matrix @ [_length(attrs.get("cx"), x1 - x0, 96, 0.5),
                              _length(attrs.get("cy"), y1 - y0, 96, 0.5), 1]
            cx, cy = point[:2]
            radius = _length(attrs.get("r"), max(x1 - x0, y1 - y0), 96, 0.5)
            radius *= math.sqrt(abs(np.linalg.det(matrix[:2, :2])))
        params[4:7] = [cx, cy, radius]
    return gradient.kind, params, gradient.stops


def _source(
    bytestring: bytes | str | None,
    file_obj,
    url: str | os.PathLike[str] | None,
) -> bytes:
    provided = sum(value is not None for value in (bytestring, file_obj, url))
    if provided != 1:
        raise TypeError("exactly one of bytestring, file_obj, or url must be provided")
    if bytestring is not None:
        return bytestring.encode() if isinstance(bytestring, str) else bytestring
    if file_obj is not None:
        data = file_obj.read()
        return data.encode() if isinstance(data, str) else data
    url_text = os.fspath(url)
    if "://" in url_text:
        with urlopen(url_text) as response:
            return response.read()
    with open(url_text, "rb") as source:
        return source.read()


class Renderer:
    def __init__(
        self,
        source: bytes,
        *,
        dpi: float,
        parent_width: float | None,
        parent_height: float | None,
        scale: float,
        output_width: int | None,
        output_height: int | None,
        background_color: str | None,
        negate_colors: bool,
        unsafe: bool,
    ):
        if not unsafe and (b"<!DOCTYPE" in source.upper() or b"<!ENTITY" in source.upper()):
            raise ValueError("DOCTYPE and entities require unsafe=True")
        self.root = ET.fromstring(source)
        if _tag(self.root) != "svg":
            raise ValueError("document root must be <svg>")
        preserve = self.root.get("preserveAspectRatio")
        if preserve and preserve.strip() not in {"xMidYMid", "xMidYMid meet"}:
            raise NotImplementedError("non-default preserveAspectRatio is not supported")
        parent_width = float(parent_width or 300)
        parent_height = float(parent_height or 150)
        viewbox_text = self.root.get("viewBox")
        viewbox = list(map(float, re.findall(_NUMBER, viewbox_text))) if viewbox_text else None
        intrinsic_width = _length(self.root.get("width"), parent_width, dpi,
                                  viewbox[2] if viewbox else 300)
        intrinsic_height = _length(self.root.get("height"), parent_height, dpi,
                                   viewbox[3] if viewbox else 150)
        target_width = intrinsic_width * scale
        target_height = intrinsic_height * scale
        if output_width is not None and output_height is not None:
            target_width, target_height = output_width, output_height
        elif output_width is not None:
            target_width, target_height = output_width, target_height * output_width / target_width
        elif output_height is not None:
            target_width, target_height = target_width * output_height / target_height, output_height
        self.width = max(1, round(target_width))
        self.height = max(1, round(target_height))
        if viewbox:
            vx, vy, vw, vh = viewbox
            factor = min(self.width / vw, self.height / vh)
            tx = (self.width - vw * factor) / 2 - vx * factor
            ty = (self.height - vh * factor) / 2 - vy * factor
            self.root_matrix = np.array([[factor, 0, tx], [0, factor, ty], [0, 0, 1.0]])
            self.user_width, self.user_height = vw, vh
        else:
            self.root_matrix = np.diag([self.width / intrinsic_width, self.height / intrinsic_height, 1.0])
            self.user_width, self.user_height = intrinsic_width, intrinsic_height
        self.dpi = dpi
        self.negate = negate_colors
        self.gradients = _definitions(self.root, negate_colors)
        self.canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        if background_color:
            background = _color(background_color, "black", False)
            self.canvas[...] = np.rint(background * 255).astype(np.uint8)

    def render(self) -> np.ndarray:
        self._walk(self.root, self.root_matrix, _DEFAULT_STYLE, 1.0)
        return self.canvas

    def _draw_rect_batch(
        self,
        element: ET.Element,
        matrix: np.ndarray,
        inherited: dict[str, str],
        parent_opacity: float,
    ) -> bool:
        children = list(element)
        if len(children) < 8 or matrix[0, 1] != 0 or matrix[1, 0] != 0:
            return False
        rects = np.empty((len(children), 8), dtype=np.float64)
        for i, child in enumerate(children):
            if (
                _tag(child) != "rect"
                or child.get("transform")
                or child.get("rx")
                or child.get("ry")
            ):
                return False
            style = _style(child, inherited)
            fill = style.get("fill", "black").strip()
            if (
                style.get("display") == "none"
                or style.get("visibility") == "hidden"
                or fill.lower() == "none"
                or _URL_PAINT.match(fill)
                or style.get("stroke", "none").strip().lower() != "none"
            ):
                return False
            x = _length(child.get("x"), self.user_width, self.dpi)
            y = _length(child.get("y"), self.user_height, self.dpi)
            width = _length(child.get("width"), self.user_width, self.dpi)
            height = _length(child.get("height"), self.user_height, self.dpi)
            x0 = matrix[0, 0] * x + matrix[0, 2]
            y0 = matrix[1, 1] * y + matrix[1, 2]
            x1 = matrix[0, 0] * (x + width) + matrix[0, 2]
            y1 = matrix[1, 1] * (y + height) + matrix[1, 2]
            color = fill
            if color == "currentColor":
                color = style.get("color", "black")
            rgba = ImageColor.getcolor(color, "RGBA")
            red, green, blue = (channel / 255.0 for channel in rgba[:3])
            if self.negate:
                red, green, blue = 1.0 - red, 1.0 - green, 1.0 - blue
            alpha = (
                rgba[3] / 255.0
                * parent_opacity
                * float(style.get("opacity", "1"))
                * float(style.get("fill-opacity", "1"))
            )
            rects[i] = x0, y0, x1, y1, red, green, blue, alpha
        draw_rects(self.canvas, rects)
        return True

    def _walk(
        self,
        element: ET.Element,
        parent_matrix: np.ndarray,
        inherited: dict[str, str],
        parent_opacity: float,
    ) -> None:
        name = _tag(element)
        style = _style(element, inherited)
        if style.get("display") == "none" or style.get("visibility") == "hidden":
            return
        if style.get("stroke-dasharray", "none").strip().lower() != "none":
            raise NotImplementedError("dashed strokes are not supported")
        if style.get("stroke-linecap", "butt") not in {"butt", "round"}:
            raise NotImplementedError("only butt and round stroke line caps are supported")
        if style.get("stroke-linejoin", "miter") != "miter":
            raise NotImplementedError("only the approximate miter stroke join is supported")
        matrix = parent_matrix @ _parse_transform(element.get("transform"))
        opacity = parent_opacity * float(style.get("opacity", "1"))
        if name == "svg" and element is not self.root:
            raise NotImplementedError("nested <svg> viewports are not supported")
        if name in {"svg", "g", "a"}:
            if self._draw_rect_batch(element, matrix, style, opacity):
                return
            for child in element:
                self._walk(child, matrix, style, opacity)
            return
        if name in {"defs", "linearGradient", "radialGradient", "stop", "title", "desc", "metadata"}:
            return
        if name in {
            "style", "text", "image", "use", "switch", "clipPath", "mask",
            "filter", "pattern",
        }:
            raise NotImplementedError(f"<{name}> is outside mojo-cairosvg's native subset")
        if name not in {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}:
            raise NotImplementedError(f"unsupported SVG element <{name}>")
        commands = _geometry(element, self.user_width, self.user_height, self.dpi)
        if not len(commands):
            return
        edges, edge_count = flatten(commands, _matrix_values(matrix))
        if not edge_count:
            return
        raw_bounds = (
            float(min(edges[0, :edge_count].min(), edges[2, :edge_count].min())),
            float(min(edges[1, :edge_count].min(), edges[3, :edge_count].min())),
            float(max(edges[0, :edge_count].max(), edges[2, :edge_count].max())),
            float(max(edges[1, :edge_count].max(), edges[3, :edge_count].max())),
        )
        scale = math.sqrt(abs(np.linalg.det(matrix[:2, :2])))
        stroke_width = _length(style.get("stroke-width"), self.user_width, self.dpi, 1.0) * scale
        padding = stroke_width / 2 + 1
        bounds = (
            math.floor(raw_bounds[0] - padding), math.floor(raw_bounds[1] - padding),
            math.ceil(raw_bounds[2] + padding), math.ceil(raw_bounds[3] + padding),
        )
        current = style.get("color", "black")
        fill = _paint(style.get("fill", "black"), current, self.gradients, raw_bounds, matrix, self.negate)
        if fill:
            kind, params, stops = fill
            draw_path(
                self.canvas, edges, edge_count, bounds, stroke_width=0, fill_rule=int(
                    style.get("fill-rule") == "evenodd"),
                round_caps=True, paint_kind=kind, paint=params, stops=stops,
                opacity=opacity * float(style.get("fill-opacity", "1")),
                samples=4, stroke=False,
            )
        stroke = _paint(style.get("stroke", "none"), current, self.gradients, raw_bounds, matrix, self.negate)
        if stroke and stroke_width > 0:
            kind, params, stops = stroke
            draw_path(
                self.canvas, edges, edge_count, bounds, stroke_width=stroke_width, fill_rule=0,
                round_caps=style.get("stroke-linecap") != "butt", paint_kind=kind,
                paint=params, stops=stops,
                opacity=opacity * float(style.get("stroke-opacity", "1")),
                samples=4, stroke=True,
            )


def render_rgba(
    bytestring=None,
    *,
    file_obj=None,
    url=None,
    dpi=96,
    parent_width=None,
    parent_height=None,
    scale=1,
    unsafe=False,
    background_color=None,
    negate_colors=False,
    invert_images=False,
    output_width=None,
    output_height=None,
) -> np.ndarray:
    del invert_images
    source = _source(bytestring, file_obj, url)
    return Renderer(
        source,
        dpi=dpi,
        parent_width=parent_width,
        parent_height=parent_height,
        scale=scale,
        output_width=output_width,
        output_height=output_height,
        background_color=background_color,
        negate_colors=negate_colors,
        unsafe=unsafe,
    ).render()


def encode_png(rgba: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()
