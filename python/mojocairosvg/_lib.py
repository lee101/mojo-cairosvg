"""ctypes bridge to the Mojo rasterizer."""

from __future__ import annotations

import ctypes
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import NoReturn

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get("MOJO_CAIROSVG_LIB", os.path.join(ROOT, "dist", "libmojo-cairosvg.so"))

I = ctypes.c_int64
F = ctypes.c_double

_lib: ctypes.CDLL | None = None
_EMPTY_STOPS = np.zeros((1, 5), dtype=np.float64)
_PARALLEL_PIXEL_THRESHOLD = 65_536
_PARALLEL_TASKS = min(16, os.cpu_count() or 1)
_executor: ThreadPoolExecutor | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB_PATH):
            result = subprocess.run(
                ["bash", os.path.join(ROOT, "build", "build.sh")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if result.returncode or not os.path.exists(LIB_PATH):
                raise RuntimeError((result.stderr or result.stdout).strip())
        library = ctypes.CDLL(LIB_PATH)
        library.mcs_flatten.argtypes = [I, I, I, I, I]
        library.mcs_flatten.restype = I
        library.mcs_draw_path.argtypes = [
            I, I, I, I, I, I, I, I, I, I, I, I, F, I, I, I, I, I, F, I
        ]
        library.mcs_draw_path.restype = None
        library.mcs_draw_rects.argtypes = [I, I, I, I, I, I]
        library.mcs_draw_rects.restype = None
        library.mcs_premultiply.argtypes = [I, I, I]
        library.mcs_premultiply.restype = None
        _lib = library
    return _lib


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_PARALLEL_TASKS)
    return _executor


def _addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null NumPy buffer to Mojo")
    return address


def _invalid(name: str, requirement: str) -> NoReturn:
    raise ValueError(f"{name} must be {requirement}")


def _input_array(
    value: np.ndarray,
    name: str,
    *,
    dtype: np.dtype,
    shape_tail: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        _invalid(name, f"{np.dtype(dtype).name}; got {array.dtype}")
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        _invalid(name, f"shape (n, {', '.join(map(str, shape_tail))})")
    return np.ascontiguousarray(array)


def _canvas_array(canvas: np.ndarray) -> np.ndarray:
    if not isinstance(canvas, np.ndarray):
        _invalid("canvas", "a NumPy array")
    if canvas.dtype != np.uint8:
        _invalid("canvas", f"uint8; got {canvas.dtype}")
    if canvas.ndim != 3 or canvas.shape[2] != 4:
        _invalid("canvas", "shape (height, width, 4)")
    if not canvas.flags.c_contiguous:
        _invalid("canvas", "C-contiguous")
    if not canvas.flags.writeable:
        _invalid("canvas", "writable")
    if canvas.size == 0:
        _invalid("canvas", "non-empty")
    return canvas


def flatten(commands: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, int]:
    commands = _input_array(
        commands, "commands", dtype=np.dtype(np.float64), shape_tail=(8,)
    )
    matrix = np.asarray(matrix)
    if matrix.dtype != np.float64:
        _invalid("matrix", f"float64; got {matrix.dtype}")
    if matrix.shape != (6,):
        _invalid("matrix", "shape (6,)")
    matrix = np.ascontiguousarray(matrix)
    if not len(commands):
        return np.empty((8, 0), dtype=np.float64), 0
    if not np.isfinite(commands).all() or not np.isfinite(matrix).all():
        raise ValueError("commands and matrix must contain only finite values")
    operations = commands[:, 0]
    if not np.all(np.isin(operations, (0.0, 1.0, 2.0, 3.0, 4.0))):
        raise ValueError("commands contains an unknown operation")
    curve_steps = commands[np.isin(operations, (2.0, 3.0)), 7]
    if np.any(curve_steps > np.iinfo(np.int64).max):
        raise OverflowError("curve step count exceeds the C ABI integer range")
    capacity = sum(
        max(2, int(row[7])) if row[0] in (2.0, 3.0) else 1
        for row in commands
    )
    edges = np.empty((8, capacity), dtype=np.float64)
    count = _load().mcs_flatten(
        _addr(commands), len(commands), _addr(matrix), _addr(edges), capacity
    )
    if count < 0 or count > capacity:
        raise RuntimeError(f"Mojo flatten returned invalid edge count {count}")
    return edges, count


def draw_path(
    canvas: np.ndarray,
    edges: np.ndarray,
    edge_count: int,
    bounds: tuple[int, int, int, int],
    *,
    stroke_width: float,
    fill_rule: int,
    round_caps: bool,
    paint_kind: int,
    paint: np.ndarray,
    stops: np.ndarray,
    opacity: float,
    samples: int,
    stroke: bool,
) -> None:
    canvas = _canvas_array(canvas)
    edges = np.asarray(edges)
    if edges.dtype != np.float64 or edges.ndim != 2 or edges.shape[0] != 8:
        _invalid("edges", "float64 with shape (8, capacity)")
    edges = np.ascontiguousarray(edges)
    if not 0 <= edge_count <= edges.shape[1]:
        raise ValueError("edge_count is outside the edges buffer")
    paint = np.asarray(paint)
    if paint.dtype != np.float64 or paint.shape != (8,):
        _invalid("paint", "float64 with shape (8,)")
    paint = np.ascontiguousarray(paint)
    stops = _input_array(
        stops, "stops", dtype=np.dtype(np.float64), shape_tail=(5,)
    )
    if not np.isfinite(edges[:, :edge_count]).all() or not np.isfinite(paint).all():
        raise ValueError("edges and paint must contain only finite values")
    if not np.isfinite(stops).all():
        raise ValueError("stops must contain only finite values")
    if samples < 1:
        raise ValueError("samples must be at least 1")
    stop_count = len(stops)
    if stop_count == 0:
        stops = _EMPTY_STOPS
    library = _load()
    x0, y0, x1, y1 = bounds
    arguments = (
        _addr(canvas), canvas.shape[1], canvas.shape[0], _addr(edges), edge_count,
        edges.shape[1], x0, int(stroke), fill_rule, stroke_width, int(round_caps),
        paint_kind, _addr(paint), _addr(stops), stop_count, opacity, samples,
    )
    top = max(0, y0)
    bottom = min(canvas.shape[0], y1)
    clipped_width = max(0, min(canvas.shape[1], x1) - max(0, x0))
    row_count = max(0, bottom - top)

    def draw_rows(row_start: int, row_end: int) -> None:
        (
            canvas_addr, width, height, edges_addr, count, stride, left,
            draw_mode, rule, line_width, caps, kind, paint_addr, stops_addr,
            stops_count, alpha, sample_count,
        ) = arguments
        library.mcs_draw_path(
            canvas_addr, width, height, edges_addr, count, stride, left,
            row_start, x1, row_end, draw_mode, rule, line_width, caps, kind,
            paint_addr, stops_addr, stops_count, alpha, sample_count,
        )

    if (
        clipped_width * row_count >= _PARALLEL_PIXEL_THRESHOLD
        and row_count > 1
        and _PARALLEL_TASKS > 1
    ):
        task_count = min(_PARALLEL_TASKS, row_count)
        rows_per_task = (row_count + task_count - 1) // task_count
        futures = [
            _pool().submit(draw_rows, start, min(start + rows_per_task, bottom))
            for start in range(top, bottom, rows_per_task)
        ]
        for future in futures:
            future.result()
    else:
        draw_rows(y0, y1)


def draw_rects(canvas: np.ndarray, rects: np.ndarray, samples: int = 4) -> None:
    canvas = _canvas_array(canvas)
    rects = _input_array(
        rects, "rects", dtype=np.dtype(np.float64), shape_tail=(8,)
    )
    if not np.isfinite(rects).all():
        raise ValueError("rects must contain only finite values")
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if not len(rects):
        return
    _load().mcs_draw_rects(
        _addr(canvas), canvas.shape[1], canvas.shape[0], _addr(rects), len(rects), samples
    )


def premultiply(rgba: np.ndarray) -> np.ndarray:
    rgba = np.asarray(rgba)
    if rgba.dtype != np.uint8:
        _invalid("rgba", f"uint8; got {rgba.dtype}")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        _invalid("rgba", "shape (height, width, 4)")
    rgba = np.ascontiguousarray(rgba)
    result = np.empty_like(rgba)
    if not rgba.size:
        return result
    _load().mcs_premultiply(_addr(rgba), _addr(result), rgba.shape[0] * rgba.shape[1])
    return result
