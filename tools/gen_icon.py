#!/usr/bin/env python
"""Generate assets/icon.ico from the icon design defined here.

`assets/icon.svg` is the reviewed SOURCE (it is what a human edits and what the
README shows). This script is the renderer: it redraws the same flat geometry
with Pillow and writes the multi-resolution `.ico` that the linker embeds.

Why not rasterize the SVG directly: the design is deliberately flat (rounded
rects + one polygon), so drawing it directly is exact and needs no SVG
dependency — no cairosvg/resvg, nothing to install, nothing to break in CI.

Usage:
    python tools/gen_icon.py            # write assets/icon.ico
    python tools/gen_icon.py --preview  # also write assets/icon-preview.png

Sizes written: 16/20/24/32/40/48/64/96/128/256. Windows picks by display
context; 16 is what a taskbar or a file list shows, so the mark must survive
it (hence the flat shapes and the wide bar, no thin strokes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "assets" / "icon.svg"
ICO = ROOT / "assets" / "icon.ico"
PREVIEW = ROOT / "assets" / "icon-preview.png"

# Design tokens. Kept in sync with icon.svg by hand — they are the same six
# numbers, and the doc comment there explains what each one is for.
BG = "#12141A"  # rounded square, near-black with a blue cast
GRID = "#3D4457"  # uncompressed tensor blocks, muted slate
ACCENT = "#E8734A"  # the collapsed/quantized bar, warm orange
SIZE = 512  # design grid; every coordinate below is in these units

# 4x4 lattice of 80-unit cells on a 112-unit margin, 24-unit gutters.
# Only rows 1-2 carry grid cells; row 3 is the collapsed bar and row 4 is
# deliberately EMPTY. Leaving live cells under the bar made two of them peek
# out below it and read as a rendering artifact rather than as intent.
CELLS = [(112, 112), (216, 112), (320, 112), (112, 216), (216, 216), (320, 216)]

# The bar that replaces row 3, in the same design units.
BAR = (98, 312, 400, 368)


def draw() -> Image.Image:
    """Render the icon at SIZE x SIZE (the base grid, no scaling)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded background. Windows draws the .ico with its own alpha; keeping a
    # rounded (not full-bleed) square avoids a black plate around the mark.
    d.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=96, fill=BG)

    # Six uncompressed blocks: rows 1-2, three each.
    for x, y in CELLS:
        d.rounded_rectangle([x, y, x + 80, y + 80], radius=14, fill=GRID)

    # Row 3 collapsed into one bar, vertically centred in the 80-unit cell
    # band so the mark stays balanced against the two grid rows above it.
    # Drawn as three overlapping shapes so the left end keeps the same rounded
    # corner as the grid cells and the right end carries a bevelled tip (the
    # README's ◆).
    x0, y0, x1, y1 = BAR
    d.rounded_rectangle([x0, y0, 140, y1], radius=14, fill=ACCENT)
    d.rectangle([112, y0, 140, y1], fill=ACCENT)
    d.polygon([(112, y0), (372, y0), (x1, 340), (372, y1), (112, y1)], fill=ACCENT)

    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="also write assets/icon-preview.png")
    args = ap.parse_args()

    assert SVG.is_file(), f"missing source design: {SVG}"

    # One 512px master; Pillow does the LANCZOS downsample for every entry in
    # `sizes`. Do NOT hand-roll a per-size redraw here — the geometry is integer
    # aligned at 512 and re-rendering small would change the mark, not sharpen it.
    base = draw()
    base.save(ICO, format="ICO", sizes=[(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)])
    print(f"wrote {ICO.relative_to(ROOT)} ({ICO.stat().st_size:,} bytes)")

    if args.preview:
        base.save(PREVIEW, format="PNG")
        print(f"wrote {PREVIEW.relative_to(ROOT)}")


if __name__ == "__main__":
    main()