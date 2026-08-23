from __future__ import annotations

import inspect
import io

import cairosvg
import numpy as np
from PIL import Image
import pytest

import mojocairosvg
import mojocairosvg._lib as native_lib
from mojocairosvg._lib import draw_path, draw_rects, flatten


def upstream(svg: bytes, **kwargs) -> np.ndarray:
    png = cairosvg.svg2png(bytestring=svg, **kwargs)
    return np.array(Image.open(io.BytesIO(png)).convert("RGBA"))


def assert_pixel_parity(svg: bytes, *, mean_error=0.8, changed_fraction=0.08, **kwargs):
    ours = mojocairosvg.svg2rgba(bytestring=svg, **kwargs)
    reference = upstream(svg, **kwargs)
    assert ours.shape == reference.shape
    difference = np.abs(ours.astype(np.int16) - reference.astype(np.int16))
    assert difference.mean() <= mean_error
    changed = np.any(difference > 8, axis=2)
    assert changed.mean() <= changed_fraction
    ours_visible = ours[..., 3] > 127
    ref_visible = reference[..., 3] > 127
    union = np.count_nonzero(ours_visible | ref_visible)
    if union:
        assert np.count_nonzero(ours_visible & ref_visible) / union > 0.90


def test_svg2png_signature_matches_upstream():
    assert inspect.signature(mojocairosvg.svg2png) == inspect.signature(cairosvg.svg2png)


def test_axis_aligned_rectangle_is_byte_identical():
    svg = b'<svg width="100" height="80"><rect x="10" y="10" width="50" height="40" fill="#f00"/></svg>'
    assert np.array_equal(mojocairosvg.svg2rgba(bytestring=svg), upstream(svg))


@pytest.mark.parametrize("element", [
    '<circle cx="50" cy="45" r="32" fill="royalblue"/>',
    '<ellipse cx="50" cy="45" rx="40" ry="23" fill="#25a060"/>',
    '<rect x="12" y="15" width="76" height="55" rx="13" ry="9" fill="orange"/>',
    '<polygon points="5,75 28,8 52,65 78,12 95,75" fill="#a030d0"/>',
])
def test_basic_filled_shapes_match_upstream(element):
    svg = f'<svg width="100" height="90">{element}</svg>'.encode()
    assert_pixel_parity(svg)


@pytest.mark.parametrize("path", [
    "M10 80 Q 60 0 110 80 Z",
    "M8 75 C 25 5 95 5 112 75 L85 52 S35 100 8 75 Z",
    "M10 70 A45 35 0 0 1 110 70 L60 20 Z",
    "M10 50 l20 -35 h45 v20 l30 30 q-40 28 -80 10 z",
])
def test_path_commands_match_upstream(path):
    svg = f'<svg width="120" height="100"><path d="{path}" fill="#20a050"/></svg>'.encode()
    assert_pixel_parity(svg, mean_error=1.2)


def test_evenodd_hole_is_byte_identical():
    svg = b"""<svg width="100" height="100">
      <path d="M5 5H95V95H5Z M30 30H70V70H30Z" fill-rule="evenodd"/>
    </svg>"""
    assert np.array_equal(mojocairosvg.svg2rgba(bytestring=svg), upstream(svg))


def test_simd_edge_tail_matches_upstream():
    points = " ".join(
        f"{60 + 48 * np.cos(i * 2 * np.pi / 19):.4f},"
        f"{60 + 48 * np.sin(i * 2 * np.pi / 19):.4f}"
        for i in range(19)
    )
    svg = (
        f'<svg width="120" height="120"><polygon points="{points}" '
        'fill="#2878c8"/></svg>'
    ).encode()
    assert_pixel_parity(svg, mean_error=0.8)


@pytest.mark.parametrize("size", [255, 256])
def test_serial_and_parallel_size_thresholds_are_exact(size):
    svg = (
        f'<svg width="{size}" height="{size}">'
        f'<rect width="{size}" height="{size}" fill="#369"/></svg>'
    ).encode()
    assert np.array_equal(mojocairosvg.svg2rgba(bytestring=svg), upstream(svg))


def test_batched_rectangles_preserve_order_and_opacity():
    shapes = "".join(
        f'<rect x="{8 + i * 7}" y="{5 + (i % 3) * 9}" width="22" height="19" '
        f'fill="#{(i * 135791 + 0x224466) % 0xFFFFFF:06x}" '
        f'fill-opacity="{0.55 + (i % 2) * 0.3}"/>'
        for i in range(9)
    )
    svg = f'<svg width="96" height="54">{shapes}</svg>'.encode()
    assert_pixel_parity(svg, mean_error=0.2, changed_fraction=0.01)


@pytest.mark.parametrize("transform", [
    "translate(20 10) rotate(20)",
    "translate(60 45) scale(1.4 .65) translate(-30 -20)",
    "matrix(1 .2 -.3 1 20 5)",
])
def test_transforms_match_upstream(transform):
    svg = f"""<svg width="130" height="100">
      <g transform="{transform}"><rect x="10" y="10" width="50" height="30" fill="#80c020"/></g>
    </svg>""".encode()
    assert_pixel_parity(svg, mean_error=1.2)


def test_viewbox_default_meet_matches_upstream():
    svg = b"""<svg width="160" height="90" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="42" fill="tomato"/>
    </svg>"""
    assert_pixel_parity(svg, mean_error=1.1)


def test_linear_gradient_matches_upstream():
    svg = b"""<svg width="140" height="80">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="red"/>
        <stop offset=".35" stop-color="#20d040" stop-opacity=".8"/>
        <stop offset="100%" stop-color="blue"/>
      </linearGradient></defs>
      <rect x="10" y="10" width="120" height="60" fill="url(#g)"/>
    </svg>"""
    assert_pixel_parity(svg, mean_error=0.8)


def test_linear_gradient_simd_tail_matches_upstream():
    svg = b"""<svg width="131" height="67">
      <defs><linearGradient id="g" x2="1" y2="1">
        <stop offset="0" stop-color="#f20"/>
        <stop offset=".45" stop-color="#2d6"/>
        <stop offset="1" stop-color="#15e"/>
      </linearGradient></defs>
      <rect width="131" height="67" fill="url(#g)"/>
    </svg>"""
    assert_pixel_parity(svg, mean_error=0.8)


def test_parallel_rows_are_byte_identical_to_serial(monkeypatch):
    svg = b"""<svg width="137" height="103"><path
      d="M3 82 C18 4 119 7 134 84 Q71 102 3 82Z"
      fill="#e87219" stroke="#253763" stroke-width="5"/></svg>"""
    monkeypatch.setattr(native_lib, "_PARALLEL_PIXEL_THRESHOLD", 1 << 60)
    serial = mojocairosvg.svg2rgba(bytestring=svg)
    monkeypatch.setattr(native_lib, "_PARALLEL_PIXEL_THRESHOLD", 1)
    parallel = mojocairosvg.svg2rgba(bytestring=svg)
    assert np.array_equal(parallel, serial)


def test_radial_gradient_matches_upstream_on_square():
    svg = b"""<svg width="100" height="100">
      <defs><radialGradient id="g">
        <stop offset="0" stop-color="white"/><stop offset="1" stop-color="navy"/>
      </radialGradient></defs>
      <rect x="10" y="10" width="80" height="80" fill="url(#g)"/>
    </svg>"""
    assert_pixel_parity(svg, mean_error=1.0)


@pytest.mark.parametrize("linecap", ["butt", "round"])
def test_stroked_path_matches_upstream(linecap):
    svg = f"""<svg width="120" height="80">
      <path d="M10 60 L60 10 L110 60" fill="none" stroke="black"
            stroke-width="7" stroke-linecap="{linecap}"/>
    </svg>""".encode()
    assert_pixel_parity(svg, mean_error=1.0)


def test_fill_then_stroke_order_and_opacity_match():
    svg = b"""<svg width="100" height="80">
      <rect x="12" y="10" width="70" height="50" fill="#e03020" fill-opacity=".65"
            stroke="#1020d0" stroke-width="8" stroke-opacity=".7"/>
    </svg>"""
    assert_pixel_parity(svg, mean_error=1.1)


def test_line_polyline_inheritance_inline_style_and_visibility():
    svg = b"""<svg width="100" height="70">
      <g fill="none" stroke="#1850a0" stroke-width="4" opacity=".8">
        <line x1="8" y1="12" x2="92" y2="12"/>
        <polyline points="8,58 30,30 55,55 90,25" style="stroke:#d04020"/>
        <line x1="0" y1="0" x2="100" y2="70" visibility="hidden"/>
        <line x1="0" y1="70" x2="100" y2="0" display="none"/>
      </g>
    </svg>"""
    assert_pixel_parity(svg, mean_error=0.8)


def test_background_color_matches_upstream():
    svg = b'<svg width="50" height="40"><circle cx="25" cy="20" r="12" fill="#00c080"/></svg>'
    assert_pixel_parity(svg, background_color="#ffeedd", mean_error=0.5)


def test_negate_colors_matches_upstream():
    svg = b'<svg width="60" height="40"><rect x="5" y="4" width="45" height="30" fill="#204080"/></svg>'
    assert_pixel_parity(svg, negate_colors=True, mean_error=0.1)


@pytest.mark.parametrize("kwargs, expected", [
    ({"scale": 2}, (80, 120)),
    ({"output_width": 90}, (60, 90)),
    ({"output_height": 25}, (25, 38)),
    ({"output_width": 90, "output_height": 30}, (30, 90)),
])
def test_output_size_semantics_match_upstream(kwargs, expected):
    svg = b'<svg width="60" height="40"><rect width="60" height="40"/></svg>'
    ours = mojocairosvg.svg2rgba(bytestring=svg, **kwargs)
    reference = upstream(svg, **kwargs)
    assert ours.shape[:2] == reference.shape[:2] == expected


def test_physical_lengths_use_dpi():
    svg = b'<svg width="1in" height="0.5in"><rect width="100%" height="100%" fill="red"/></svg>'
    assert mojocairosvg.svg2rgba(bytestring=svg, dpi=120).shape == (60, 120, 4)


def test_parent_dimensions_resolve_root_percentages():
    svg = b'<svg width="50%" height="25%"><rect width="100%" height="100%"/></svg>'
    assert mojocairosvg.svg2rgba(
        bytestring=svg, parent_width=200, parent_height=120
    ).shape == (30, 100, 4)


def test_file_object_and_path_inputs(tmp_path):
    svg = b'<svg width="12" height="8"><rect width="12" height="8" fill="red"/></svg>'
    path = tmp_path / "input.svg"
    path.write_bytes(svg)
    from_file = mojocairosvg.svg2png(file_obj=io.BytesIO(svg))
    from_path = mojocairosvg.svg2png(url=path)
    assert from_file == from_path == mojocairosvg.svg2png(bytestring=svg)


def test_write_to_file_object_and_path(tmp_path):
    svg = b'<svg width="12" height="8"><circle cx="6" cy="4" r="3"/></svg>'
    buffer = io.BytesIO()
    assert mojocairosvg.svg2png(bytestring=svg, write_to=buffer) is None
    path = tmp_path / "result.png"
    assert mojocairosvg.svg2png(bytestring=svg, write_to=path) is None
    assert path.read_bytes() == buffer.getvalue()
    assert Image.open(path).size == (12, 8)


def test_exactly_one_input_is_required():
    with pytest.raises(TypeError, match="exactly one"):
        mojocairosvg.svg2png()
    with pytest.raises(TypeError, match="exactly one"):
        mojocairosvg.svg2png(bytestring=b"<svg/>", file_obj=io.BytesIO())


def test_unsafe_controls_doctype_rejection():
    svg = b'<!DOCTYPE svg><svg width="1" height="1"/>'
    with pytest.raises(ValueError, match="DOCTYPE"):
        mojocairosvg.svg2png(bytestring=svg)
    assert mojocairosvg.svg2png(bytestring=svg, unsafe=True).startswith(b"\x89PNG")


def test_unsupported_text_fails_explicitly():
    svg = b'<svg width="100" height="20"><text x="0" y="15">hello</text></svg>'
    with pytest.raises(NotImplementedError, match="<text>"):
        mojocairosvg.svg2png(bytestring=svg)


@pytest.mark.parametrize("attribute", [
    'stroke-dasharray="2 2"',
    'stroke-linecap="square"',
    'stroke-linejoin="bevel"',
])
def test_unsupported_stroke_features_fail_explicitly(attribute):
    svg = f'<svg width="20" height="20"><path d="M1 1L19 19" stroke="black" {attribute}/></svg>'.encode()
    with pytest.raises(NotImplementedError):
        mojocairosvg.svg2png(bytestring=svg)


def test_nondefault_preserve_aspect_ratio_fails_explicitly():
    svg = b'<svg width="20" height="10" viewBox="0 0 10 10" preserveAspectRatio="none"/>'
    with pytest.raises(NotImplementedError, match="preserveAspectRatio"):
        mojocairosvg.svg2png(bytestring=svg)


@pytest.mark.parametrize("content", [
    '<svg x="1" y="1" width="10" height="10"/>',
    '<switch><rect width="10" height="10"/></switch>',
    '<defs><linearGradient id="g" gradientTransform="rotate(20)"/></defs>',
    '<defs><radialGradient id="g" fx=".2"/></defs>',
])
def test_other_document_features_outside_subset_fail_explicitly(content):
    svg = f'<svg width="20" height="20">{content}</svg>'.encode()
    with pytest.raises(NotImplementedError):
        mojocairosvg.svg2png(bytestring=svg)


def test_premultiply_alpha_matches_integer_reference():
    rng = np.random.default_rng(4)
    rgba = rng.integers(0, 256, size=(73, 91, 4), dtype=np.uint8)
    wide = rgba.astype(np.uint16)
    wide[..., :3] = (wide[..., :3] * wide[..., 3:4] + 127) // 255
    assert np.array_equal(mojocairosvg.premultiply_alpha(rgba), wide.astype(np.uint8))


def test_premultiply_alpha_validates_shape():
    with pytest.raises(ValueError, match="shape"):
        mojocairosvg.premultiply_alpha(np.zeros((10, 4), dtype=np.uint8))


def test_premultiply_alpha_rejects_silent_dtype_narrowing():
    with pytest.raises(ValueError, match="uint8"):
        mojocairosvg.premultiply_alpha(np.zeros((2, 3, 4), dtype=np.float64))


def test_flatten_curve_capacity_matches_mojo_minimum_steps():
    commands = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1],
        [2, 1, 0, 2, 1, 3, 1, 0],
    ], dtype=np.float64)
    edges, count = flatten(commands, np.eye(3, dtype=np.float64)[[0, 1], :].T.ravel())
    assert edges.shape == (8, 3)
    assert count == 2


@pytest.mark.parametrize("bad", [
    np.zeros((2, 8), dtype=np.float32),
    np.zeros((2, 7), dtype=np.float64),
])
def test_flatten_rejects_unsafe_command_buffers(bad):
    with pytest.raises(ValueError):
        flatten(bad, np.array([1, 0, 0, 1, 0, 0], dtype=np.float64))


def test_draw_wrappers_reject_invalid_output_layout_and_lengths():
    canvas = np.zeros((4, 5, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="C-contiguous"):
        draw_rects(canvas[:, ::-1], np.zeros((1, 8), dtype=np.float64))

    edges = np.zeros((8, 1), dtype=np.float64)
    paint = np.zeros(8, dtype=np.float64)
    with pytest.raises(ValueError, match="edge_count"):
        draw_path(
            canvas, edges, 2, (0, 0, 1, 1), stroke_width=0, fill_rule=0,
            round_caps=True, paint_kind=0, paint=paint,
            stops=np.empty((0, 5), dtype=np.float64), opacity=1, samples=4,
            stroke=False,
        )
