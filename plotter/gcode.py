"""
Turns the model's generated strokes into G-code for the CNC plotter.
Only the ML strokes are converted, never the human's. Everything is in plotter mm.
"""

# Machine settings. Placeholders until the plotter is calibrated.
# Positions are mm from the pen's home spot (GRBL's X0 Y0); +X and +Y move away from it.
PAPER_BOUNDS = (0, 0, 297, 210)  # (xmin, ymin, xmax, ymax) of the paper, A4 landscape for now
FLIP_Y = True                    # True if +Y moves toward the top of the page (canvas y points down)
FLIP_X = False                   # True if +X moves toward the left of the page
FEED = 1500                      # drawing speed, mm/min
RAPID_RATE = 3000                # pen-up speed, mm/min; match GRBL $110/$111 (only used for time estimates)
PEN_UP_CMD = "M5"                # servo commands for a GRBL servo fork; Marlin uses "M280 P0 S<angle>"
PEN_DOWN_CMD = "M3 S90"          # TODO: calibrate S values on the actual servo
PEN_DELAY = 0.15                 # seconds to wait for the servo to finish moving
HOME = (0, 0)
PORT = None                      # Arduino serial port, e.g. "COM7"; None finds it automatically
BAUD = 115200                    # GRBL default

# Pen index constants (same layout as the model's stroke-5 vectors)
PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2


# Maps canvas pixels (origin top-left, y down) onto the paper in mm.
# The canvas is scaled to fit inside the paper without stretching, and centred.
class CanvasToPaper:
    def __init__(self, canvas_w, canvas_h, bounds=PAPER_BOUNDS, flip_x=FLIP_X, flip_y=FLIP_Y):
        xmin, ymin, xmax, ymax = bounds
        self.scale = min((xmax - xmin) / canvas_w, (ymax - ymin) / canvas_h)   # mm per pixel
        self.x0 = xmin + ((xmax - xmin) - canvas_w * self.scale) / 2
        self.y0 = ymin + ((ymax - ymin) - canvas_h * self.scale) / 2
        self.canvas_w, self.canvas_h = canvas_w, canvas_h
        self.flip_x, self.flip_y = flip_x, flip_y

    # Absolute canvas point -> (x, y) mm
    def point(self, x, y):
        if self.flip_x:
            x = self.canvas_w - x
        if self.flip_y:
            y = self.canvas_h - y
        return (self.x0 + x * self.scale, self.y0 + y * self.scale)

    # Stroke-5 deltas in pixels -> the same strokes in mm (pen states unchanged)
    def strokes(self, strokes):
        sx = -self.scale if self.flip_x else self.scale
        sy = -self.scale if self.flip_y else self.scale
        return [[s[0] * sx, s[1] * sy] + list(s[2:]) for s in strokes]


# strokes:  [[dx, dy, penDown, penUp, penEnd], ...] from model.sample(), in mm (see CanvasToPaper)
# anchor:   (x, y) mm, end of the human's last pen-down line. The first delta is measured from here.
# bounds:   (xmin, ymin, xmax, ymax) mm, the drawable area of the paper
# Returns a list of G-code lines
def strokes_to_gcode(
    strokes,
    anchor,
    bounds=PAPER_BOUNDS,
    feed=FEED,                  # drawing speed, mm/min
    pen_up_cmd=PEN_UP_CMD,
    pen_down_cmd=PEN_DOWN_CMD,
    pen_delay=PEN_DELAY,        # seconds to wait for the servo to finish moving
    home=HOME,
    max_steps=250,              # the model doesn't always emit penEnd
):
    # The servo moves on its own and the firmware doesn't wait for it,
    # so pause after every pen command or the pen drags/skips at stroke starts.
    # G4 P is seconds on GRBL but milliseconds on Marlin.
    dwell = f"G4 P{pen_delay}"

    lines = ["G21", "G90", pen_up_cmd, dwell]   # mm, absolute coordinates, pen up to be safe

    x, y = anchor              # running absolute position (never clipped)
    prev_pen_down = False      # model's pen state; the human just lifted their pen
    pen_is_down = False        # where the physical pen actually is
    prev_in_bounds = True      # the anchor is where the human drew, so it's on the paper

    # Travel from home to where the human stopped
    lines.append(f"G0 X{x:.2f} Y{y:.2f}")

    for i, stroke in enumerate(strokes):
        if i >= max_steps:
            break

        dx, dy = stroke[0], stroke[1]
        x += dx
        y += dy

        # Clip only the printed copy so later points don't shift
        cx = min(max(x, bounds[0]), bounds[2])
        cy = min(max(y, bounds[1]), bounds[3])
        in_bounds = (cx == x and cy == y)

        # The previous stroke's pen state decides whether this move draws.
        # Both ends must be on the paper, or we'd draw a stray line from a clipped edge point.
        draw = prev_pen_down and in_bounds and prev_in_bounds
        prev_in_bounds = in_bounds

        # Only send pen commands when the state changes
        if draw and not pen_is_down:
            lines += [pen_down_cmd, dwell]
            pen_is_down = True
        elif not draw and pen_is_down:
            lines += [pen_up_cmd, dwell]
            pen_is_down = False

        if draw:
            lines.append(f"G1 X{cx:.2f} Y{cy:.2f} F{feed}")
        else:
            lines.append(f"G0 X{cx:.2f} Y{cy:.2f}")

        if stroke[2 + PEN_END] == 1:
            break

        # Applies to the NEXT move; a failed sample ([0, 0, 0]) counts as pen up
        prev_pen_down = (stroke[2 + PEN_DOWN] == 1)

    lines += [pen_up_cmd, dwell, f"G0 X{home[0]:.2f} Y{home[1]:.2f}"]
    return lines


def write_gcode(lines, path):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    fake = [
        [10, 0, 1, 0, 0],    # pen-up travel (previous state was up), then pen down
        [0, 10, 1, 0, 0],    # draws
        [-10, 0, 0, 1, 0],   # draws, then lifts
        [-5, -5, 1, 0, 0],   # pen-up travel
        [5, 0, 0, 0, 1],     # draws, then end
    ]
    gcode = strokes_to_gcode(fake, anchor=(50, 50), bounds=(0, 0, 200, 150))
    print("\n".join(gcode))
    write_gcode(gcode, "test.gcode")


if __name__ == "__main__":
    main()
