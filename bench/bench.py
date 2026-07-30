"""End-to-end CairoSVG comparisons and one isolated pixel kernel."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import cairosvg
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojocairosvg as mojo  # noqa: E402


def timeit(function, repeats=3):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def svg_cases():
    solid = b'<svg width="768" height="768"><rect width="768" height="768" fill="#369"/></svg>'
    gradient = b"""<svg width="768" height="768"><defs><linearGradient id="g" x2="1" y2="1">
      <stop stop-color="#f20"/><stop offset=".5" stop-color="#2d6"/>
      <stop offset="1" stop-color="#15e"/></linearGradient></defs>
      <rect width="768" height="768" fill="url(#g)"/></svg>"""
    path = b"""<svg width="384" height="384"><path
      d="M8 300 C35 12 345 12 376 300 Q230 380 8 300Z"
      fill="#ef9020" stroke="#273060" stroke-width="5"/></svg>"""
    shapes = "".join(
        f'<rect x="{(i * 17) % 490}" y="{(i * 31) % 490}" width="18" height="13" '
        f'fill="#{(i * 123457) % 0xFFFFFF:06x}"/>'
        for i in range(500)
    )
    many = f'<svg width="512" height="512">{shapes}</svg>'.encode()
    return [
        ("solid fill, 768x768", solid),
        ("linear gradient, 768x768", gradient),
        ("curved fill + stroke, 384x384", path),
        ("500 small rectangles, 512x512", many),
    ]


def main():
    print(f"Machine: {cpu_name()}, {platform.system()} {platform.machine()}")
    print(f"Python {platform.python_version()}, CairoSVG {cairosvg.__version__}")
    print()
    print("| case | Mojo | reference | speedup | result |")
    print("|---|---:|---:|---:|---|")
    for name, svg in svg_cases():
        ours = lambda: mojo.svg2png(bytestring=svg)
        theirs = lambda: cairosvg.svg2png(bytestring=svg)
        ours()
        theirs()
        mojo_time = timeit(ours)
        reference_time = timeit(theirs)
        ratio = reference_time / mojo_time
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| svg2png: {name} | {mojo_time * 1000:.2f} ms | "
            f"{reference_time * 1000:.2f} ms | {ratio:.2f}x | {result} |"
        )

    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, size=(2048, 2048, 4), dtype=np.uint8)

    def numpy_premultiply():
        wide = rgba.astype(np.uint16)
        wide[..., :3] = (wide[..., :3] * wide[..., 3:4] + 127) // 255
        return wide.astype(np.uint8)

    ours = lambda: mojo.premultiply_alpha(rgba)
    ours()
    numpy_premultiply()
    mojo_time = timeit(ours, repeats=5)
    reference_time = timeit(numpy_premultiply, repeats=5)
    ratio = reference_time / mojo_time
    result = "faster" if ratio >= 1 else "slower"
    print(
        f"| premultiply RGBA, 2048x2048 | {mojo_time * 1000:.2f} ms | "
        f"{reference_time * 1000:.2f} ms (NumPy) | {ratio:.2f}x | {result} |"
    )


if __name__ == "__main__":
    main()
