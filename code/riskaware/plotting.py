"""Dependency-light plotting with Pillow.

The bundled runtime has no matplotlib, and the paper's final figures will be
rebuilt from real result tables anyway.  These helpers exist so that the sanity
run and the result-table templates can be visualised on any machine, and so
that every sanity figure carries a visible "SYNTHETIC" watermark.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (31, 91, 168),
    (214, 94, 33),
    (58, 143, 79),
    (150, 60, 150),
    (120, 120, 120),
    (200, 40, 60),
]


def font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - older Pillow
        return ImageFont.load_default()


def _axes(draw, box, xlim, ylim, xlabel, ylabel, f):
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=1)
    # y ticks
    for i in range(5):
        yy = y1 - (y1 - y0) * i / 4
        val = ylim[0] + (ylim[1] - ylim[0]) * i / 4
        draw.line([x0 - 4, yy, x0, yy], fill=(0, 0, 0))
        draw.text((x0 - 8, yy), f"{val:.2f}", fill=(0, 0, 0), font=f, anchor="rm")
    for i in range(5):
        xx = x0 + (x1 - x0) * i / 4
        val = xlim[0] + (xlim[1] - xlim[0]) * i / 4
        draw.line([xx, y1, xx, y1 + 4], fill=(0, 0, 0))
        draw.text((xx, y1 + 6), f"{val:.2f}", fill=(0, 0, 0), font=f, anchor="mt")
    draw.text(((x0 + x1) / 2, y1 + 26), xlabel, fill=(0, 0, 0), font=f, anchor="mt")


def _watermark(img: Image.Image, text: str) -> None:
    d = ImageDraw.Draw(img)
    f = font(13)
    d.text((14, img.height - 24), text, fill=(190, 40, 40), font=f)


def line_plot(
    series: Dict[str, Sequence[Tuple[float, float]]],
    path: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    note: str = "SYNTHETIC PIPELINE SANITY CHECK - NOT A RESULT",
    size: Tuple[int, int] = (980, 580),
) -> str:
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font(14)
    ft = font(17)
    if title:
        d.text((size[0] / 2, 14), title, fill=(0, 0, 0), font=ft, anchor="mt")
    box = (90, 60, size[0] - 230, size[1] - 90)
    xs = [p[0] for s in series.values() for p in s]
    ys = [p[1] for s in series.values() for p in s]
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    xlim = (min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1.0)
    ylim = (min(0.0, min(ys)), max(ys) * 1.08 if max(ys) > 0 else 1.0)
    _axes(d, box, xlim, ylim, xlabel, ylabel, f)
    x0, y0, x1, y1 = box

    def sx(v):
        return x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)

    def sy(v):
        return y1 - (v - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)

    for i, (name, pts) in enumerate(series.items()):
        col = PALETTE[i % len(PALETTE)]
        seq = sorted(pts)
        for a, b in zip(seq, seq[1:]):
            d.line([sx(a[0]), sy(a[1]), sx(b[0]), sy(b[1])], fill=col, width=2)
        for p in seq:
            d.ellipse([sx(p[0]) - 3, sy(p[1]) - 3, sx(p[0]) + 3, sy(p[1]) + 3], fill=col)
        lx, ly = size[0] - 220, 70 + 22 * i
        d.line([lx, ly, lx + 22, ly], fill=col, width=2)
        d.text((lx + 30, ly), name, fill=(0, 0, 0), font=f, anchor="lm")
    _watermark(img, note)
    img.save(path)
    return path


def timeline_plot(
    rows: Dict[str, Sequence[float]],
    t: Sequence[float],
    path: str,
    title: str = "",
    note: str = "SYNTHETIC PIPELINE SANITY CHECK - NOT A RESULT",
    size: Tuple[int, int] = (1000, 620),
) -> str:
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font(13)
    ft = font(16)
    if title:
        d.text((size[0] / 2, 12), title, fill=(0, 0, 0), font=ft, anchor="mt")
    n = len(rows)
    top, left, right = 110, 110, size[0] - 40
    panel_h = (size[1] - top - 60) / max(1, n)
    t0, t1 = (min(t), max(t)) if t else (0.0, 1.0)
    for i, (name, vals) in enumerate(rows.items()):
        y_base = top + panel_h * (i + 1) - 12
        d.line([left, y_base, right, y_base], fill=(180, 180, 180))
        d.text((left - 10, y_base - panel_h / 2), name, fill=(0, 0, 0), font=f, anchor="rm")
        hi = max(1e-6, max(vals)) if vals else 1.0
        pts = []
        for ti, v in zip(t, vals):
            xx = left + (ti - t0) / max(1e-9, t1 - t0) * (right - left)
            yy = y_base - (v / hi) * (panel_h - 22)
            pts.append((xx, yy))
        for a, b in zip(pts, pts[1:]):
            d.line([a[0], a[1], b[0], b[1]], fill=PALETTE[i % len(PALETTE)], width=2)
    _watermark(img, note)
    img.save(path)
    return path
