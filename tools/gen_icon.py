#!/usr/bin/env python
"""Generate assets/icon.ico from the design in assets/icon.svg.

`assets/icon.svg` is the reviewed SOURCE (it is what a human edits and what the
README shows). This script is the renderer: it PARSES that file and rasterizes
the shapes it finds, so the two cannot drift apart.

That matters because the pair already drifted once. The previous version
hard-coded the geometry in Python and merely asserted the SVG existed, so
"edit the SVG, then regenerate" was a lie — regenerating silently reverted the
design. Reading the SVG makes the documented workflow true by construction.

Parsing is a deliberate subset, not a general SVG implementation: the icon is
flat geometry only (a background rect, a group of small rects, one 4-point
polygon), so a full renderer — and the cairosvg/resvg dependency it would drag
in — would buy nothing.

Usage:
    python tools/gen_icon.py            # write assets/icon.ico
    python tools/gen_icon.py --preview  # also write assets/icon-preview.png

Sizes written: 16/20/24/32/40/48/64/96/128/256. Windows picks by display
context; 16 is what a taskbar or a file list shows, so the mark must survive
it. The funnel's fine cells necessarily mush at 16px (a cell is 1.5% of the
width), so the silhouette is what carries there.

Needs Pillow. It is NOT in this project's managed interpreter; use
C:/WinApp/Dev/Python314/python.exe or the user venv at C:/_Dev/Python/.venv.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "assets" / "icon.svg"
ICO = ROOT / "assets" / "icon.ico"
PREVIEW = ROOT / "assets" / "icon-preview.png"

SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
SVG_NS = "{http://www.w3.org/2000/svg}"
PLATE_COLOR = "#12141A"


def _rgb(color: str) -> tuple[int, int, int, int]:
    """`#RRGGBB` (the only form the icon uses) -> RGBA."""
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"only #RRGGBB is supported, got {color!r}")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255)


def _numbers(text: str) -> list[float]:
    return [float(t) for t in re.findall(r"-?\d*\.?\d+", text)]


class Design:
    """The icon's shapes in 512-unit model space, straight from the SVG."""

    def __init__(self) -> None:
        root = ET.parse(SVG).getroot()
        if not root.tag.endswith("svg"):
            raise ValueError(f"{SVG} is not an SVG file")

        self.plate: tuple[float, float, float, float, float] | None = None
        self.ink: tuple[int, int, int, int] | None = None
        self.cells: list[tuple[float, float, float, float]] = []
        self.wedge: list[tuple[float, float]] | None = None

        for el in root.iter():
            tag = el.tag.replace(SVG_NS, "")
            if tag == "rect":
                fill = el.get("fill")
                w = float(el.get("width", 0))
                # The full-bleed rounded square is the plate; the rest are ink
                # cells. Distinguish by geometry, not by tree position, so
                # reordering the SVG cannot silently swap their roles.
                if el.get("rx") and fill and w >= 512:
                    self.plate = (
                        float(el.get("x", 0)),
                        float(el.get("y", 0)),
                        w,
                        float(el.get("height")),
                        float(el.get("rx")),
                    )
                else:
                    self.cells.append(
                        (
                            float(el.get("x", 0)),
                            float(el.get("y", 0)),
                            w,
                            float(el.get("height")),
                        )
                    )
                    self.ink = self.ink or _rgb(fill or "")
            elif tag == "g" and el.get("fill"):
                self.ink = self.ink or _rgb(el.get("fill"))
            elif tag == "path":
                pts = _numbers(el.get("d", ""))
                poly = list(zip(pts[0::2], pts[1::2]))
                if len(poly) < 3:
                    raise ValueError("wedge path needs at least 3 points")
                self.wedge = poly
                self.ink = self.ink or _rgb((el.get("fill") or "").strip())

        missing = [
            name
            for name, ok in (
                ("plate", self.plate is not None),
                ("ink colour", self.ink is not None),
                ("cells", bool(self.cells)),
                ("wedge", self.wedge is not None),
            )
            if not ok
        ]
        if missing:
            raise ValueError(f"{SVG.name} is missing: {', '.join(missing)}")

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [c[0] for c in self.cells] + [p[0] for p in self.wedge]
        ys = [c[1] for c in self.cells] + [p[1] for p in self.wedge]
        xs += [c[0] + c[2] for c in self.cells]
        ys += [c[1] + c[3] for c in self.cells]
        return min(xs), min(ys), max(xs), max(ys)

    def draw(self, px: int) -> Image.Image:
        """Rasterize at `px` x `px`.

        ONE transform (k = px/512), applied uniformly at draw time. Do not also
        scale the model coordinates while parsing — that double-applies the
        scale and walks the mark off the plate, which is exactly the bug the
        first draft of this script had.
        """
        k = px / 512.0
        img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        x, y, w, h, r = self.plate
        d.rounded_rectangle(
            [x * k, y * k, (x + w) * k - 1, (y + h) * k - 1],
            radius=r * k,
            fill=PLATE_COLOR,
        )
        for cx, cy, cw, ch in self.cells:
            d.rectangle([cx * k, cy * k, (cx + cw) * k, (cy + ch) * k], fill=self.ink)
        d.polygon([(a * k, b * k) for a, b in self.wedge], fill=self.ink)
        return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="also write assets/icon-preview.png")
    args = ap.parse_args()

    if not SVG.is_file():
        raise SystemExit(f"missing source design: {SVG}")

    design = Design()
    x0, y0, x1, y1 = design.bounds()
    print(f"mark bbox: {x0:.1f},{y0:.1f} -> {x1:.1f},{y1:.1f}  (must sit inside 0..512)")

    # Render EVERY size directly rather than letting Pillow downsample one
    # 512px master: a cell is 19.34/512 = 3.8% of the width, so at 16px it is
    # 0.6px. Downsampling merges neighbours into a smear; drawing at the final
    # scale lets the wedge and the coarse block stay crisp where they exist.
    base = design.draw(512)
    base.save(ICO, format="ICO", sizes=[(s, s) for s in SIZES])
    print(
        f"wrote {ICO.relative_to(ROOT)} ({ICO.stat().st_size:,} bytes) "
        f"from {len(design.cells)} cells + 1 wedge, {len(SIZES)} sizes"
    )

    if args.preview:
        base.save(PREVIEW, format="PNG")
        print(f"wrote {PREVIEW.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
