"""
Virtual plotter: runs G-code without the hardware so we can check it first.

Reads the G-code roughly the way GRBL would, flags anything that would go wrong on the real
machine, estimates how long the drawing takes, and draws or animates the result.
The plot shows the page the way the person drawing sees it.

    python plotter/simulate.py file.gcode              # show the drawing
    python plotter/simulate.py file.gcode --animate    # watch it draw
    python plotter/simulate.py file.gcode --out a.png  # save a picture (.gif saves the animation)
    python plotter/simulate.py --test                  # run the self-tests
"""
import argparse
import bisect
import math
import re
import sys
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotter.gcode import (FLIP_X, FLIP_Y, HOME, PAPER_BOUNDS, PEN_DOWN_CMD, PEN_UP_CMD,
                           RAPID_RATE, CanvasToPaper, strokes_to_gcode)

WORD = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")
EPS = 1e-6


# One thing the machine does: a move, or a pause when start == end
@dataclass
class Step:
    x0: float
    y0: float
    x1: float
    y1: float
    pen_down: bool
    seconds: float

    @property
    def moves(self):
        return abs(self.x1 - self.x0) > EPS or abs(self.y1 - self.y0) > EPS


def strip_comment(line):
    return re.sub(r"\(.*?\)", "", line).split(";")[0].strip()


# "G1 X10 F1500" -> (("F", 1500.0), ("G", 1.0), ("X", 10.0)), sorted so equal commands compare equal
def _words(line):
    return tuple(sorted((l, float(v)) for l, v in WORD.findall(strip_comment(line).upper())))


# Runs G-code lines through a simple model of a GRBL pen plotter.
# The machine starts at X0 Y0 with the pen up, like after power-on.
# Returns {"steps": [Step], "warnings": [str], "stats": {...}}
def simulate(lines, bounds=PAPER_BOUNDS, pen_up_cmd=PEN_UP_CMD, pen_down_cmd=PEN_DOWN_CMD,
             rapid_rate=RAPID_RATE):
    down_key, up_key = _words(pen_down_cmd), _words(pen_up_cmd)
    xmin, ymin, xmax, ymax = bounds

    def on_paper(x, y):
        return xmin - EPS <= x <= xmax + EPS and ymin - EPS <= y <= ymax + EPS

    x = y = 0.0
    pen_down = False
    absolute = True
    motion = None           # modal G0 / G1
    feed = None
    waiting_dwell = None    # line of a pen command not yet followed by G4
    steps = []
    stats = {"draw_mm": 0.0, "travel_mm": 0.0, "pen_lowers": 0}

    # Each problem is reported once: message -> [first line, count]
    found = {}
    def warn(n, msg):
        found.setdefault(msg, [n, 0])[1] += 1

    for n, raw in enumerate(lines, 1):
        words = _words(raw)
        if not words:
            continue
        codes = [(l, int(v)) for l, v in words if l in "GM"]
        params = {l: v for l, v in words if l not in "GM"}

        if waiting_dwell is not None and ("G", 4) not in codes:
            warn(waiting_dwell, "pen command not followed by a G4 pause, so the pen may still be moving")
        waiting_dwell = None

        # Pen up / down
        new_pen = None
        if words in (down_key, up_key):
            new_pen = words == down_key
        elif ("M", 3) in codes or ("M", 4) in codes or ("M", 5) in codes:
            warn(n, f"pen command differs from the configured ones ({pen_down_cmd!r} / {pen_up_cmd!r})")
            new_pen = ("M", 5) not in codes
        if new_pen is not None:
            if new_pen and not pen_down:
                stats["pen_lowers"] += 1
            pen_down = new_pen
            waiting_dwell = n
            continue

        for code in codes:
            if code in (("G", 0), ("G", 1)):
                motion = code[1]
            elif code == ("G", 90):
                absolute = True
            elif code == ("G", 91):
                absolute = False
            elif code == ("G", 20):
                warn(n, "G20 (inches) isn't supported, only mm")
            elif code == ("G", 4):
                steps.append(Step(x, y, x, y, pen_down, params.get("P", 0.0)))
            elif code not in (("G", 21), ("M", 2), ("M", 30)):
                warn(n, f"{code[0]}{code[1]} isn't simulated")

        if "F" in params:
            feed = params["F"]
        if "Z" in params:
            warn(n, "Z moves are ignored (the pen is lifted by the servo)")

        if "X" not in params and "Y" not in params:
            continue
        if motion is None:
            warn(n, "X/Y given before any G0/G1")
            continue

        if absolute:
            nx, ny = params.get("X", x), params.get("Y", y)
        else:
            nx, ny = x + params.get("X", 0.0), y + params.get("Y", 0.0)
        dist = math.hypot(nx - x, ny - y)

        if motion == 1 and feed is None:
            warn(n, "G1 before any feed rate (F); GRBL rejects this with error:22")
        rate = (feed or rapid_rate) if motion == 1 else rapid_rate

        if pen_down:
            if motion == 0:
                warn(n, "G0 rapid move with the pen down, so it draws at full speed")
            if not (on_paper(x, y) and on_paper(nx, ny)):
                warn(n, "draws outside the paper")
            stats["draw_mm"] += dist
        else:
            stats["travel_mm"] += dist

        steps.append(Step(x, y, nx, ny, pen_down, dist / rate * 60))
        x, y = nx, ny

    if pen_down:
        warn(len(lines), "file ends with the pen still down")

    stats["seconds"] = sum(s.seconds for s in steps)
    warnings = [f"line {n}: {msg}" + (f" ({count} times)" if count > 1 else "")
                for msg, (n, count) in found.items()]
    return {"steps": steps, "warnings": warnings, "stats": stats}


def summary(sim):
    s = sim["stats"]
    lowers = s["pen_lowers"]
    return (f"Draws {s['draw_mm']:.0f} mm and travels {s['travel_mm']:.0f} mm with the pen up, "
            f"lowering the pen {lowers} time{'' if lowers == 1 else 's'}. "
            f"About {s['seconds']:.0f} s, not counting acceleration.")


def _setup_axes(ax, sim, bounds, flip_x, flip_y):
    xmin, ymin, xmax, ymax = bounds
    xs = [xmin, xmax, HOME[0]] + [v for s in sim["steps"] for v in (s.x0, s.x1)]
    ys = [ymin, ymax, HOME[1]] + [v for s in sim["steps"] for v in (s.y0, s.y1)]
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys))

    ax.set_facecolor("0.88")
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor="white", edgecolor="0.4", label="paper"))
    ax.plot(*HOME, "+", color="tab:blue", ms=10, mew=2, label="home (X0 Y0)")
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    # Show the page the way the person drawing sees it
    if flip_x:
        ax.invert_xaxis()
    if not flip_y:
        ax.invert_yaxis()


def _travel_lines(segments):
    return LineCollection(segments, colors="0.55", linewidths=0.8, linestyles=(0, (2, 2)),
                          label="pen up (travel)")


def _drawn_lines(segments):
    return LineCollection(segments, colors="black", linewidths=1.6, label="pen down (draws)")


# A still picture of the whole job. overlay: optional human lines in mm, [[(x, y), ...], ...]
def render(sim, bounds=PAPER_BOUNDS, overlay=None, ax=None, flip_x=FLIP_X, flip_y=FLIP_Y):
    if ax is None:
        ax = Figure(figsize=(7, 5)).add_subplot()
    _setup_axes(ax, sim, bounds, flip_x, flip_y)

    moves = [s for s in sim["steps"] if s.moves]
    if overlay:
        ax.add_collection(LineCollection(overlay, colors="tab:red", linewidths=1.6, label="human"))
    ax.add_collection(_travel_lines([[(s.x0, s.y0), (s.x1, s.y1)] for s in moves if not s.pen_down]))
    ax.add_collection(_drawn_lines([[(s.x0, s.y0), (s.x1, s.y1)] for s in moves if s.pen_down]))
    ax.legend(loc="upper right", fontsize=8)
    return ax.figure


# Plays the job back in about `seconds`, keeping the machine's order and timing proportions
def animate(sim, bounds=PAPER_BOUNDS, overlay=None, seconds=10, fps=30, flip_x=FLIP_X, flip_y=FLIP_Y):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    _setup_axes(ax, sim, bounds, flip_x, flip_y)
    if overlay:
        ax.add_collection(LineCollection(overlay, colors="tab:red", linewidths=1.6, label="human"))
    travel, drawn = _travel_lines([]), _drawn_lines([])
    ax.add_collection(travel)
    ax.add_collection(drawn)
    pen, = ax.plot([], [], "o", ms=9, mec="tab:orange", mew=2, label="pen")
    ax.legend(loc="upper right", fontsize=8)

    steps = sim["steps"]
    ends = list(accumulate(s.seconds for s in steps))   # when each step finishes
    total = ends[-1] if ends else 0.0
    frames = max(2, int(seconds * fps))

    def update(i):
        t = total * i / (frames - 1)
        done = bisect.bisect_right(ends, t)     # steps finished by time t
        segs = [s for s in steps[:done] if s.moves]
        d = [[(s.x0, s.y0), (s.x1, s.y1)] for s in segs if s.pen_down]
        u = [[(s.x0, s.y0), (s.x1, s.y1)] for s in segs if not s.pen_down]

        if done < len(steps):                   # the step in progress
            s = steps[done]
            start = ends[done] - s.seconds
            f = (t - start) / s.seconds if s.seconds > 0 else 1.0
            px, py = s.x0 + f * (s.x1 - s.x0), s.y0 + f * (s.y1 - s.y0)
            (d if s.pen_down else u).append([(s.x0, s.y0), (px, py)])
            down = s.pen_down
        elif steps:
            px, py, down = steps[-1].x1, steps[-1].y1, steps[-1].pen_down
        else:
            px, py, down = HOME[0], HOME[1], False

        drawn.set_segments(d)
        travel.set_segments(u)
        pen.set_data([px], [py])
        pen.set_markerfacecolor("tab:orange" if down else "none")
        ax.set_title(f"{t:5.1f} s of {total:.1f} s machine time, pen {'down' if down else 'up'}", fontsize=10)
        return drawn, travel, pen

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)
    return fig, anim


def self_test():
    results = []
    def check(name, ok):
        print("PASS" if ok else "FAIL", name)
        results.append(ok)
    def close(a, b):
        return abs(a - b) < 1e-6

    # Neeya's demo strokes from gcode.py
    box = (0, 0, 200, 150)
    fake = [
        [10, 0, 1, 0, 0],
        [0, 10, 1, 0, 0],
        [-10, 0, 0, 1, 0],
        [-5, -5, 1, 0, 0],
        [5, 0, 0, 0, 1],
    ]
    sim = simulate(strokes_to_gcode(fake, anchor=(50, 50), bounds=box), bounds=box)
    check("demo G-code has no warnings", sim["warnings"] == [])
    check("demo draws 3 lines", sum(s.moves and s.pen_down for s in sim["steps"]) == 3)
    check("demo draws 25 mm", close(sim["stats"]["draw_mm"], 25))
    check("demo lowers the pen twice", sim["stats"]["pen_lowers"] == 2)

    # Mistakes the real machine would hit
    bad = simulate(["G21", "G90", "M3 S90", "G0 X10 Y0", "G1 X400 Y0", "M5", "G4 P0.1",
                    "M3 S90", "G4 P0.1", "G1 X20 Y20 F1000"], bounds=box)
    text = "\n".join(bad["warnings"])
    check("catches a missing G4 after a pen command", "line 3: pen command not followed" in text)
    check("catches a rapid move with the pen down", "line 4: G0 rapid move" in text)
    check("catches G1 with no feed rate", "line 5: G1 before any feed rate" in text)
    check("catches drawing off the paper", "line 5: draws outside the paper (2 times)" in text)
    check("catches ending with the pen down", "pen still down" in text)

    # Canvas pixels -> paper mm, 700x400 canvas onto A4 landscape
    a4 = (0, 0, 297, 210)
    m = CanvasToPaper(700, 400, bounds=a4, flip_y=True)
    y0 = (210 - 400 * 297 / 700) / 2
    check("canvas fills the paper's width", close(m.scale, 297 / 700))
    check("canvas bottom-left -> paper bottom-left", all(map(close, m.point(0, 400), (0, y0))))
    check("canvas top-right -> paper top-right", all(map(close, m.point(700, 0), (297, 210 - y0))))
    for fx, fy in ((False, True), (True, False)):
        flipped = CanvasToPaper(700, 400, bounds=a4, flip_x=fx, flip_y=fy)
        p, q = flipped.point(100, 50), flipped.point(130, 90)
        d = flipped.strokes([[30, 40, 1, 0, 0]])[0]
        check(f"deltas agree with points (flip_x={fx}, flip_y={fy})",
              close(p[0] + d[0], q[0]) and close(p[1] + d[1], q[1]) and d[2:] == [1, 0, 0])

    # Round trip: canvas strokes -> G-code -> simulator ends where the canvas says
    px_strokes = [[30, 0, 1, 0, 0], [0, 40, 1, 0, 0], [-30, 0, 0, 1, 0]]
    g = strokes_to_gcode(m.strokes(px_strokes), m.point(100, 100), bounds=a4)
    sim = simulate(g, bounds=a4)
    last = [s for s in sim["steps"] if s.moves and s.pen_down][-1]
    check("round trip has no warnings", sim["warnings"] == [])
    ex, ey = m.point(100, 140)
    check("round trip ends at canvas (100, 140), to G-code's 0.01 mm",
          abs(last.x1 - ex) <= 0.005 and abs(last.y1 - ey) <= 0.005)

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


def main():
    p = argparse.ArgumentParser(description="Simulate a pen plotter G-code file.")
    p.add_argument("file", nargs="?", help=".gcode file to simulate")
    p.add_argument("--animate", action="store_true", help="watch it draw")
    p.add_argument("--out", help="save to .png (picture) or .gif (animation) instead of showing a window")
    p.add_argument("--test", action="store_true", help="run the self-tests")
    a = p.parse_args()

    if a.test:
        sys.exit(self_test())
    if not a.file:
        p.error("give a .gcode file, or --test")

    sim = simulate(Path(a.file).read_text().splitlines())
    print(summary(sim))
    print(f"{len(sim['warnings'])} problem(s):" if sim["warnings"] else "No problems found.")
    for w in sim["warnings"]:
        print("  " + w)

    if a.out and a.out.lower().endswith(".gif"):
        fig, anim = animate(sim)
        anim.save(a.out, writer="pillow", fps=30, dpi=80)
    elif a.out:
        render(sim).savefig(a.out, dpi=150, bbox_inches="tight")
    else:
        import matplotlib.pyplot as plt
        if a.animate:
            fig, anim = animate(sim)    # keep `anim` alive or the animation stops
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            render(sim, ax=ax)
        plt.show()


if __name__ == "__main__":
    main()
