"""CairoSVG-compatible entry points backed by a native Mojo rasterizer."""

from __future__ import annotations

import os
import numpy as np

from ._svg import encode_png, render_rgba
from ._lib import premultiply

__version__ = "0.1.0"


def svg2png(
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
    write_to=None,
    output_width=None,
    output_height=None,
):
    """Convert the supported SVG subset to PNG.

    The signature and bytes/file/path behavior match ``cairosvg.svg2png``.
    """
    rgba = render_rgba(
        bytestring,
        file_obj=file_obj,
        url=url,
        dpi=dpi,
        parent_width=parent_width,
        parent_height=parent_height,
        scale=scale,
        unsafe=unsafe,
        background_color=background_color,
        negate_colors=negate_colors,
        invert_images=invert_images,
        output_width=output_width,
        output_height=output_height,
    )
    png = encode_png(rgba)
    if write_to is None:
        return png
    if hasattr(write_to, "write"):
        write_to.write(png)
    else:
        with open(os.fspath(write_to), "wb") as destination:
            destination.write(png)
    return None


def svg2rgba(
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
):
    """Render directly to an ``(height, width, 4)`` uint8 array."""
    return render_rgba(
        bytestring,
        file_obj=file_obj,
        url=url,
        dpi=dpi,
        parent_width=parent_width,
        parent_height=parent_height,
        scale=scale,
        unsafe=unsafe,
        background_color=background_color,
        negate_colors=negate_colors,
        invert_images=invert_images,
        output_width=output_width,
        output_height=output_height,
    )


def premultiply_alpha(rgba):
    """Return uint8 RGBA with RGB multiplied by alpha in the Mojo kernel."""
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError("rgba must have shape (height, width, 4)")
    return premultiply(array)


__all__ = ["svg2png", "svg2rgba", "premultiply_alpha", "__version__"]
