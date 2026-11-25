"""Copy the blue drawing visible on the blue-dot-calibrated paper.

Run from the repository root:

    python CV/test_copy.py --dry-run   # trace and simulate; motors do not move
    python CV/test_copy.py             # trace, then press P to send to the plotter

The test is intentionally interactive so detecting a noisy mask can never start
the machine by itself. Put the plotter pen at HOME before pressing P.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CV.server import (CanvasCalibrator, draw_calibration, get_canvas_transform,
                       get_ink_mask)
from HW.Plotter.gcode import CanvasToPaper, strokes_to_gcode, write_gcode


OUTPUT_DIR = ROOT / "HW" / "Plotter" / "output"
GCODE_PATH = OUTPUT_DIR / "test_copy.gcode"
PREVIEW_PATH = OUTPUT_DIR / "test_copy_preview.png"


def skeletonize(mask):
    """Reduce thick ink blobs to a one-pixel centreline with Zhang-Suen thinning."""
    image = (mask > 0).astype(np.uint8)

    def neighbours():
        padded = np.pad(image, 1)
        return [
            padded[:-2, 1:-1],   # north
            padded[:-2, 2:],     # north-east
            padded[1:-1, 2:],    # east
            padded[2:, 2:],      # south-east
            padded[2:, 1:-1],    # south
            padded[2:, :-2],     # south-west
            padded[1:-1, :-2],   # west
            padded[:-2, :-2],    # north-west
        ]

    changed = True
    while changed:
        changed = False
        for first_pass in (True, False):
            around = neighbours()
            count = sum(around)
            transitions = sum(
                (around[index] == 0) & (around[(index + 1) % 8] == 1)
                for index in range(8)
            )
            north, _, east, _, south, _, west, _ = around
            if first_pass:
                preserve_a = north * east * south
                preserve_b = east * south * west
            else:
                preserve_a = north * east * west
                preserve_b = north * south * west
            remove = ((image == 1) & (count >= 2) & (count <= 6)
                      & (transitions == 1) & (preserve_a == 0) & (preserve_b == 0))
            if np.any(remove):
                image[remove] = 0
                changed = True
    return image * 255


def _edge(a, b):
    return (a, b) if a <= b else (b, a)


def skeleton_to_paths(skeleton, simplify=2.0, minimum_length=8):
    """Turn skeleton pixels into polylines, splitting at ends and junctions."""
    ys, xs = np.nonzero(skeleton)
    points = {(int(x), int(y)) for x, y in zip(xs, ys)}
    if not points:
        return []

    def neighbours(point):
        x, y = point
        found = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidate = (x + dx, y + dy)
                if not (dx or dy) or candidate not in points:
                    continue
                # At a staircase corner, the diagonal is a redundant shortcut
                # that would create a false three-way junction.
                if dx and dy and ((x + dx, y) in points or (x, y + dy) in points):
                    continue
                found.append(candidate)
        return found

    adjacent = {point: neighbours(point) for point in points}
    visited = set()
    raw_paths = []

    def trace(start, second):
        path = [start, second]
        visited.add(_edge(start, second))
        previous, current = start, second
        while len(adjacent[current]) == 2:
            following = adjacent[current][0] if adjacent[current][1] == previous else adjacent[current][1]
            edge = _edge(current, following)
            if edge in visited:
                break
            visited.add(edge)
            path.append(following)
            previous, current = current, following
        return path

    # Trace open branches first, beginning at endpoints and junctions.
    nodes = [point for point in points if len(adjacent[point]) != 2]
    for start in nodes:
        for second in adjacent[start]:
            if _edge(start, second) not in visited:
                raw_paths.append(trace(start, second))

    # Any edges left belong to closed loops, which have no endpoint.
    for start in points:
        for second in adjacent[start]:
            if _edge(start, second) not in visited:
                raw_paths.append(trace(start, second))

    paths = []
    for path in raw_paths:
        length = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
        if length < minimum_length:
            continue
        contour = np.asarray(path, np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(contour, simplify, False).reshape(-1, 2)
        if len(simplified) >= 2:
            paths.append([(float(x), float(y)) for x, y in simplified])
    return paths


def find_drawing_paths(frame):
    """Extract drawable centreline paths from a rectified BGR paper image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = get_ink_mask(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # Calibration dots map to the image corners. Remove a 5 mm border so they
    # cannot become tiny copy paths if part of a dot remains inside the crop.
    border = 10
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask, skeleton_to_paths(skeletonize(mask))


def paths_to_strokes(paths):
    """Convert absolute polylines to the stroke-5 format used by Streamlit."""
    anchor = paths[0][0]
    current = anchor
    strokes = []
    for path in paths:
        # Travel to a new path with the pen up. This movement also sets the
        # state that makes the following movement lower the pen and draw.
        start = path[0]
        strokes.append([start[0] - current[0], start[1] - current[1], 1, 0, 0])
        current = start
        for index, point in enumerate(path[1:], 1):
            is_last = index == len(path) - 1
            strokes.append([
                point[0] - current[0], point[1] - current[1],
                0 if is_last else 1,
                1 if is_last else 0,
                0,
            ])
            current = point
    strokes.append([0, 0, 0, 0, 1])
    return anchor, strokes


def paths_to_gcode(paths, canvas_size):
    """Use the same CanvasToPaper -> strokes_to_gcode flow as Streamlit."""
    width, height = canvas_size
    to_paper = CanvasToPaper(width, height)
    anchor, strokes = paths_to_strokes(paths)
    return strokes_to_gcode(
        to_paper.strokes(strokes),
        to_paper.point(*anchor),
        max_steps=len(strokes),
    )


def draw_preview(frame, paths):
    preview = frame.copy()
    for path in paths:
        points = np.rint(path).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [points], False, (0, 0, 255), 2)
    cv2.putText(preview, "RED = COPY PATH | P = plot | R = recapture | Q = quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return preview


def capture_drawing(camera):
    calibrator = CanvasCalibrator()
    transform = None
    canvas_size = None
    window = "Copy test - SPACE to capture, Q to quit"
    while True:
        ok, raw_frame = camera.read()
        if not ok:
            raise RuntimeError("The camera stopped returning frames")

        corners = calibrator.update(raw_frame)
        if corners is not None:
            transform, canvas_size = get_canvas_transform(corners)

        if transform is None:
            rectified = None
            shown = raw_frame.copy()
            draw_calibration(shown, calibrator.samples[-1] if calibrator.samples else None)
            label = "Yellow rings = recognized dots"
        else:
            rectified = cv2.warpPerspective(raw_frame, transform, canvas_size)
            shown = rectified.copy()
            label = "SPACE = capture drawing | Q = quit"

        label_y = 65 if transform is None else 30
        cv2.putText(shown, label, (10, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow(window, shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return None, None
        if key == ord(" ") and transform is not None:
            return rectified, canvas_size


def main():
    # Importing the sender also loads the plotter simulator and matplotlib. Keep
    # that work out of unit-test discovery, which imports every test_*.py file.
    from HW.Plotter.send import PORT as default_port
    from HW.Plotter.send import FakeGrbl, PlotterError, send_gcode

    parser = argparse.ArgumentParser(
        description="Trace the blue board drawing and copy it with the plotter."
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--port", default=default_port, help="plotter serial port")
    parser.add_argument("--dry-run", action="store_true", help="use a pretend plotter")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {args.camera}")

    try:
        while True:
            frame, canvas_size = capture_drawing(camera)
            if frame is None:
                return
            mask, paths = find_drawing_paths(frame)
            if not paths:
                print("No blue drawing was found. Press SPACE to try another frame.")
                continue

            preview = draw_preview(frame, paths)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(PREVIEW_PATH), preview)
            gcode = paths_to_gcode(paths, canvas_size)
            write_gcode(gcode, GCODE_PATH)
            print(f"Found {len(paths)} path(s). Preview: {PREVIEW_PATH}")
            print(f"G-code: {GCODE_PATH}")
            cv2.imshow("Copy preview", cv2.hconcat([
                preview, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            ]))

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    return
                if key in (ord("r"), ord("R")):
                    break
                if key in (ord("p"), ord("P")):
                    print("Dry run: no motors will move." if args.dry_run else
                          "Starting plot. The pen must already be at the HOME corner.")
                    send_gcode(
                        gcode,
                        ser=FakeGrbl() if args.dry_run else None,
                        port=args.port,
                        on_progress=lambda i, n, line: print(f"[{i}/{n}] {line}"),
                    )
                    print("Copy test complete.")
                    return
    except PlotterError as error:
        sys.exit(f"Stopped: {error}")
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
