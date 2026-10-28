"""
Makes test G-code files in plotter/output from drawings where we know what the right answer looks like.

    python plotter/make_samples.py

sample_ruler.gcode        a 50 mm square, a 40 mm circle and a 100 mm ruler, drawn directly in mm.
                          Measure them on the paper to check the plotter's steps per mm.
sample_cat_face.gcode     a hand-made cat face on the 700x400 canvas, sent through the same
                          pixel -> mm conversion as the app. Checks the conversion and the flip.
sample_model_cats.gcode   six cats the model draws from scratch, to see what the model itself can do.
"""
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.sketch_support import line_to_stroke
from plotter.gcode import PAPER_BOUNDS, CanvasToPaper, strokes_to_gcode, write_gcode
from plotter.simulate import render, simulate, summary

CANVAS_W, CANVAS_H = 700, 400       # same as UI/canvas.py
OUT = Path(__file__).resolve().parent / "output"


def circle(cx, cy, r, n=48):
    angles = (2 * math.pi * i / n for i in range(n + 1))
    return [[cx + r * math.cos(a), cy + r * math.sin(a)] for a in angles]


# Polylines (in mm, or in canvas px with a CanvasToPaper) -> G-code.
# Each polyline is drawn, then the pen lifts and travels to the next one.
def polylines_to_gcode(lines, to_paper=None):
    anchor = lines[0][0]
    strokes, last = [], anchor
    for line in lines:
        strokes += line_to_stroke(line, last)
        last = line[-1]
    if to_paper:
        strokes, anchor = to_paper.strokes(strokes), to_paper.point(*anchor)
    return strokes_to_gcode(strokes, anchor, max_steps=len(strokes))


def ruler():
    x0, y0 = PAPER_BOUNDS[0] + 20, PAPER_BOUNDS[1] + 20
    square = [[x0, y0], [x0 + 50, y0], [x0 + 50, y0 + 50], [x0, y0 + 50], [x0, y0]]
    ring = circle(x0 + 100, y0 + 25, 20, 72)
    rule = [[x0, y0 + 80], [x0 + 100, y0 + 80]]
    ticks = [[[x0 + 10 * i, y0 + 80], [x0 + 10 * i, y0 + 80 + (8 if i % 5 == 0 else 4)]] for i in range(11)]
    return polylines_to_gcode([square, ring, rule] + ticks)


def cat_face():
    def mirror(line):
        return [[CANVAS_W - x, y] for x, y in line]

    head = circle(350, 210, 110, 64)
    ear = [[255, 155], [245, 45], [295, 115]]
    eye = circle(310, 185, 12, 16)
    nose = [[350, 235], [338, 222], [362, 222], [350, 235]]
    mouth = [[350, 235], [350, 248], [336, 260], [322, 252]]
    whiskers = [[[300, 235], [200, 215]], [[300, 245], [195, 248]], [[300, 255], [205, 282]]]

    left = [ear, eye, mouth] + whiskers
    lines = [head, nose] + left + [mirror(l) for l in left]
    return polylines_to_gcode(lines, CanvasToPaper(CANVAS_W, CANVAS_H))


# The model draws one sketch from scratch. Returns polylines in canvas px, starting at (0, 0).
def model_draw(model, temperature, max_steps=250):
    state = model.zeroState()
    inp = [0, 0, 1, 0, 0]               # sketch-rnn's start token
    x = y = 0.0
    lines, cur = [], [[x, y]]
    for _ in range(max_steps):
        state = model.update(inp, state)
        s = model.sample(model.getPDF(state, temperature))
        if s[4] == 1:                   # pen_end: finished
            break
        x, y = x + s[0], y + s[1]
        if inp[2] == 1:                 # previous pen state down: this move draws
            cur.append([x, y])
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [[x, y]]
        inp = s
    if len(cur) > 1:
        lines.append(cur)
    return lines


# Six model cats in a 3x2 grid on the canvas, each shrunk to fit its cell if needed
def model_cats(temperature=0.2, seed=0, cols=3, rows=2, margin=15):
    from ML.model import SketchRNN

    random.seed(seed)
    model = SketchRNN("url")
    model.initialize()
    model.setPixelFactor(5.0)           # same as the app

    cell_w, cell_h = CANVAS_W / cols, CANVAS_H / rows
    lines = []
    for i in range(cols * rows):
        print(f"  model drawing cat {i + 1} of {cols * rows}")
        cat = model_draw(model, temperature)
        xs = [p[0] for l in cat for p in l]
        ys = [p[1] for l in cat for p in l]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        shrink = min(1, (cell_w - 2 * margin) / max(w, 1e-9), (cell_h - 2 * margin) / max(h, 1e-9))
        cx, cy = (i % cols + 0.5) * cell_w, (i // cols + 0.5) * cell_h    # cell centre
        mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2          # cat centre
        lines += [[[cx + (px - mx) * shrink, cy + (py - my) * shrink] for px, py in l] for l in cat]
    return polylines_to_gcode(lines, CanvasToPaper(CANVAS_W, CANVAS_H))


def main():
    OUT.mkdir(exist_ok=True)
    samples = {"sample_ruler": ruler(), "sample_cat_face": cat_face()}
    print("Loading the model (takes a few seconds)...")
    samples["sample_model_cats"] = model_cats()

    for name, gcode in samples.items():
        write_gcode(gcode, OUT / f"{name}.gcode")
        sim = simulate(gcode)
        render(sim).savefig(OUT / f"{name}.png", dpi=120, bbox_inches="tight")
        print(f"\n{name}.gcode (+ .png): {summary(sim)}")
        for w in sim["warnings"]:
            print("  problem:", w)
    print(f"\nSaved in {OUT}")


if __name__ == "__main__":
    main()
