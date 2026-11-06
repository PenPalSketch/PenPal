"""Convert SketchRNN stroke-3 sequences to fixed-size bitmaps.

Stroke-3 format: array of (dx, dy, pen_state) where pen_state == 1 means
"this is the last point of the current stroke; pen lifts before the next move".
Use the same rasterizer at train time AND at comparison time, otherwise the
classifier will measure rasterizer differences rather than drawing quality.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


IMG_SIZE = 64
PAD = 4
LINE_WIDTH = 2


def strokes_to_polylines(strokes: np.ndarray) -> list[list[tuple[float, float]]]:
    strokes = np.asarray(strokes)
    polylines: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [(0.0, 0.0)]
    x = y = 0.0
    for dx, dy, pen in strokes:
        x += float(dx)
        y += float(dy)
        current.append((x, y))
        if int(pen) == 1:
            if len(current) > 1:
                polylines.append(current)
            current = [(x, y)]
    if len(current) > 1:
        polylines.append(current)
    return polylines


def rasterize(
    strokes: np.ndarray,
    size: int = IMG_SIZE,
    pad: int = PAD,
    line_width: int = LINE_WIDTH,
) -> np.ndarray:
    """Return an (size, size) uint8 image: 0 = background, 255 = ink."""
    polylines = strokes_to_polylines(strokes)
    if not polylines:
        return np.zeros((size, size), dtype=np.uint8)

    pts = np.array([p for line in polylines for p in line], dtype=np.float32)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = float((max_xy - min_xy).max())
    if span <= 0:
        return np.zeros((size, size), dtype=np.uint8)

    draw_size = size - 2 * pad
    scale = draw_size / span
    extent = (max_xy - min_xy) * scale
    center_offset = (draw_size - extent) / 2.0

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for line in polylines:
        coords = [
            (
                (x - min_xy[0]) * scale + pad + center_offset[0],
                (y - min_xy[1]) * scale + pad + center_offset[1],
            )
            for (x, y) in line
        ]
        draw.line(coords, fill=255, width=line_width)
    return np.asarray(img, dtype=np.uint8)


def rasterize_batch(strokes_list, size: int = IMG_SIZE) -> np.ndarray:
    out = np.zeros((len(strokes_list), size, size), dtype=np.uint8)
    for i, s in enumerate(strokes_list):
        out[i] = rasterize(s, size=size)
    return out


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sketchrnn_cat.npz"
    data = np.load(path, allow_pickle=True, encoding="latin1")
    sample = data["train"][0]
    img = rasterize(sample)
    print(f"strokes shape: {sample.shape}, image shape: {img.shape}, ink pixels: {(img > 0).sum()}")
    Image.fromarray(img).save("preview.png")
    print("wrote preview.png")
