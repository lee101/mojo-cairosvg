# mojo-cairosvg

`mojo-cairosvg` is a native Mojo rasterizer for a useful, deliberately bounded
subset of [CairoSVG](https://cairosvg.org/). It keeps SVG document preparation
in Python and moves the compute-heavy work—curve flattening, fill and stroke
coverage, gradient sampling, alpha compositing, and pixel premultiplication—into
one compiled Mojo shared library.

The package is standalone at runtime: it does not call CairoSVG or Cairo.
CairoSVG is a development dependency in `pixi.toml` so every supported feature
can be pixel-tested against the real upstream implementation.

## Coverage

The public `mojocairosvg.svg2png` function has the same signature as
`cairosvg.svg2png`, including byte strings, file objects and paths; DPI,
parent-size and output-size controls; background and color negation; and
`write_to` behavior. `svg2rgba` returns the rendered `uint8` array directly, and
`premultiply_alpha` exposes the native RGBA pixel kernel.

The native SVG subset includes:

- `path` with absolute and relative `M/L/H/V/C/S/Q/T/A/Z` commands
- `rect` (including rounded corners), `circle`, `ellipse`, `line`, `polyline`,
  and `polygon`
- nested `g` containers, inherited presentation attributes, inline
  `style`, transforms, element/fill/stroke opacity, and `display`/`visibility`
- solid fills and strokes, nonzero and even-odd fill rules, butt and round line
  caps
- linear and centered radial gradients with arbitrary color stops
- intrinsic dimensions, physical units, percentages, `viewBox`, default
  `xMidYMid meet` fitting, scaling, and explicit output dimensions

The following are not covered: nested `svg` viewport semantics, CSS stylesheets
and selectors, text and fonts,
embedded images, `use`, clipping, masks, filters, patterns, markers, dashed
strokes, square caps, exact stroke joins, gradient transforms/spread methods and
focal radial gradients, or non-default
`preserveAspectRatio`. PDF, PS and SVG output entry points are also outside this
rasterization-focused port. Unsupported elements raise `NotImplementedError`
instead of silently rendering an incomplete document.

Antialiasing is a four-by-four coverage grid in a one-pixel edge band. It is
not Cairo's antialiaser, so boundary alpha values can differ slightly even when
the visible geometry agrees. The test suite checks full RGBA error, changed
pixel fraction and visible-mask overlap against CairoSVG 2.9.0.

## Install

Install the pinned Mojo nightly and all Python dependencies, then build the
shared library:

```bash
pixi install
pixi run build
```

Run verification and benchmarks with:

```bash
pixi run test
pixi run bench
```

The benchmark task holds a machine-wide file lock so concurrent jobs do not
distort the result.

## Usage

```python
import mojocairosvg as cairosvg

svg = b"""
<svg width="240" height="120" viewBox="0 0 240 120">
  <defs>
    <linearGradient id="sky">
      <stop offset="0" stop-color="#4ea3ff"/>
      <stop offset="1" stop-color="#183a78"/>
    </linearGradient>
  </defs>
  <rect width="240" height="120" rx="14" fill="url(#sky)"/>
  <path d="M20 95 Q75 25 125 90 T220 62"
        fill="none" stroke="white" stroke-width="6" stroke-linecap="round"/>
</svg>
"""

png_bytes = cairosvg.svg2png(bytestring=svg, output_width=480)
with open("result.png", "wb") as output:
    output.write(png_bytes)
```

For an in-memory pixel array:

```python
rgba = cairosvg.svg2rgba(bytestring=svg)
premultiplied = cairosvg.premultiply_alpha(rgba)
```

## Benchmark

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, CairoSVG 2.9.0. Times are the best of three runs
(five for premultiplication), after warm-up. Speedup is reference time divided
by Mojo time.

| case | Mojo | reference | speedup | result |
|---|---:|---:|---:|---|
| svg2png: solid fill, 768x768 | 12.04 ms | 25.47 ms | 2.12x | faster |
| svg2png: linear gradient, 768x768 | 31.09 ms | 42.32 ms | 1.36x | faster |
| svg2png: curved fill + stroke, 384x384 | 22.69 ms | 11.33 ms | 0.50x | slower |
| svg2png: 500 small rectangles, 512x512 | 24.59 ms | 61.26 ms | 2.49x | faster |
| premultiply RGBA, 2048x2048 | 22.96 ms | 268.55 ms (NumPy) | 11.70x | faster |

In this run, solid fill, the gradient, the rectangle batch, and the
premultiplication kernel were faster. The curved-path case remained slower.
Large path bounds are split into independent native row ranges above a
65,536-pixel threshold; smaller draws stay serial. Opaque linear-gradient spans
use float64 SIMD with scalar boundary and remainder handling.

No GPU path is included. Gradient painting has low arithmetic intensity, while
path coverage is branch-heavy and repeatedly scans a small edge buffer; neither
kernel is a good fit for GPU transfer and launch overhead.

## How it works

Python parses XML, resolves inherited presentation attributes, converts shape
elements into path commands, and computes the document and element transform
matrices. A single call flattens each path's quadratic, cubic and elliptical-arc
segments into screen-space line edges in Mojo.

The renderer stores the destination as C-contiguous, straight-alpha RGBA8 in
row-major order. For each shape, Mojo limits work to its pixel bounding box,
classifies pixels away from an edge with one center sample, supersamples only
the boundary band, evaluates solid or gradient paint, and composites it
source-over into the destination.

The Python boundary uses `ctypes`. NumPy buffers cross the C ABI as 64-bit
integer addresses; exported Mojo functions reconstruct
`UnsafePointer[..., AnyOrigin[mut=True]]` values internally. Nothing owns memory
on both sides: Python allocates every command, edge and pixel buffer, and Mojo
only reads or mutates those buffers during the call. Large draws use a bounded
Python worker pool; each ctypes call releases the GIL and owns disjoint canvas
rows. `src/capi.mojo` is one compilation unit and `build/build.sh` emits
`dist/libmojo-cairosvg.so`.
