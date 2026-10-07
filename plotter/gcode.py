"""
Turns the model's generated strokes into G-code for the CNC plotter.
Only the ML strokes are converted, never the human's. Everything is in plotter mm.
"""

# Pen index constants (same layout as the model's stroke-5 vectors)
PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2

# strokes:  [[dx, dy, penDown, penUp, penEnd], ...] from model.sample()
# anchor:   (x, y) mm, end of the human's last pen-down line. The first delta is measured from here.
# bounds:   (xmin, ymin, xmax, ymax) mm, the drawable area of the paper
# Returns a list of G-code lines
def strokes_to_gcode(
    strokes,
    anchor,
    bounds,
    feed=1500,              # drawing speed, mm/min
    pen_up_cmd="M5",        # servo commands for a GRBL servo fork; Marlin uses "M280 P0 S<angle>"
    pen_down_cmd="M3 S90",  # TODO: calibrate S values on the actual servo
    pen_delay=0.15,         # seconds to wait for the servo to finish moving
    home=(0, 0),
    max_steps=250,          # the model doesn't always emit penEnd
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
