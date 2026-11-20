import asyncio
import json
import math
import platform
import time
from collections import deque

import cv2
import numpy as np
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from util import largest_contour

def find_canvas_corners(hsv, min_area_fraction=0.1):
    # The canvas shows up gray on camera, so tell it from the brown wood by saturation,
    # not brightness: canvas saturation is ~10-20, the wood ~190-230.
    white = cv2.inRange(hsv, (0, 0, 90), (180, 80, 255))
    kernel = np.ones((9, 9), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
    contour = largest_contour(white)
    height, width = hsv.shape[:2]
    if contour is None or cv2.contourArea(contour) < min_area_fraction * width * height:
        return None

    hull = cv2.convexHull(contour)
    corners = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(corners) != 4:
        corners = cv2.boxPoints(cv2.minAreaRect(contour))
    return order_corners(corners.astype(np.float32))


def order_corners(corners):
    # Top-left has the smallest x + y, bottom-right the largest;
    # top-right has the smallest y - x, bottom-left the largest.
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).ravel()
    return np.array([
        corners[np.argmin(sums)], corners[np.argmin(diffs)],
        corners[np.argmax(sums)], corners[np.argmax(diffs)],
    ], dtype=np.float32)


class CanvasCalibrator:
    def __init__(self, duration_seconds=1.5):
        self.duration_seconds = duration_seconds
        self.start = None
        self.samples = []

    def update(self, hsv):
        """Return the canvas corners once enough frames have been seen, otherwise None."""
        if self.start is None:
            self.start = time.monotonic()
        corners = find_canvas_corners(hsv)
        if corners is not None:
            self.samples.append(corners)
        # Keep looking past the deadline until the canvas has been seen at least once.
        if time.monotonic() - self.start < self.duration_seconds or not self.samples:
            return None
        return np.median(self.samples, axis=0).astype(np.float32)


def get_canvas_transform(corners):
    top_left, top_right, bottom_right, bottom_left = corners
    width = int(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    height = int(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, target), (width, height)


def draw_calibration(frame, corners):
    if corners is not None:
        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.putText(frame, "FINDING CANVAS - keep it clear", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def get_ink_mask(hsv):
    return cv2.inRange(hsv, (95, 60, 40), (130, 255, 255))


def get_arm_mask(hsv):
    # The canvas is white, so anything saturated or dark is the arm, except the blue ink.
    not_white = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 60, 0), (180, 255, 255)),
        cv2.inRange(hsv, (0, 0, 0), (180, 255, 90)),
    )
    mask = cv2.bitwise_and(not_white, cv2.bitwise_not(get_ink_mask(hsv)))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def get_arm(hsv, min_area_fraction=0.01):
    contour = largest_contour(get_arm_mask(hsv))
    height, width = hsv.shape[:2]
    if contour is None or cv2.contourArea(contour) < min_area_fraction * width * height:
        return None, None

    # The arm enters from a frame edge, so the hand is the point farthest from that edge.
    points = contour.reshape(-1, 2)
    margin = 5
    on_edge = points[
        (points[:, 0] < margin) | (points[:, 0] >= width - margin)
        | (points[:, 1] < margin) | (points[:, 1] >= height - margin)
    ]
    if len(on_edge) == 0:
        moments = cv2.moments(contour)
        return contour, (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    entry = on_edge.mean(axis=0)
    tip = points[np.argmax(np.linalg.norm(points - entry, axis=1))]
    return contour, (int(tip[0]), int(tip[1]))


DIRECTIONS = ["RIGHT", "DOWN-RIGHT", "DOWN", "DOWN-LEFT", "LEFT", "UP-LEFT", "UP", "UP-RIGHT"]


class ArmDirectionTracker:
    def __init__(self, window_seconds=0.5, min_motion=25):
        self.window_seconds = window_seconds
        self.min_motion = min_motion
        self.history = deque()

    def update(self, tip):
        now = time.monotonic()
        if tip is None:
            self.history.clear()
            return None, None
        self.history.append((now, tip))
        while now - self.history[0][0] > self.window_seconds:
            self.history.popleft()

        (_, start), (_, end) = self.history[0], self.history[-1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        if math.hypot(dx, dy) < self.min_motion:
            return "STILL", None
        # Image y grows downward, so a positive dy is DOWN.
        sector = round(math.atan2(dy, dx) / (math.pi / 4)) % 8
        return DIRECTIONS[sector], (dx, dy)


def draw_arm(frame, contour, tip, direction, motion):
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (255, 128, 0), 2)
        cv2.circle(frame, tip, 6, (255, 0, 255), -1)
        if motion is not None:
            length = math.hypot(*motion)
            end = (int(tip[0] + motion[0] / length * 60), int(tip[1] + motion[1] / length * 60))
            cv2.arrowedLine(frame, tip, end, (255, 0, 255), 3, tipLength=0.3)
    label = f"ARM: {direction or 'NONE'}"
    cv2.putText(frame, label, (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


def geodesic_distance(mask, seed):
    """Distance from seed to every pixel of mask, measured along the mask (-1 where unreachable)."""
    distance = np.full(mask.shape, -1, np.int32)
    distance[seed[1], seed[0]] = 0
    queue = deque([(seed[1], seed[0])])
    height, width = mask.shape
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and distance[ny, nx] < 0:
                    distance[ny, nx] = distance[y, x] + 1
                    queue.append((ny, nx))
    return distance


def nearest_pixel(mask, point):
    ys, xs = np.nonzero(mask)
    nearest = np.argmin((xs - point[0]) ** 2 + (ys - point[1]) ** 2)
    return int(xs[nearest]), int(ys[nearest])


def farthest_pixel(distance):
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    return int(x), int(y)


def find_endpoints(mask):
    # The pixel farthest along the line from anywhere is one end; the one farthest from that is the other.
    ys, xs = np.nonzero(mask)
    first_end = farthest_pixel(geodesic_distance(mask, (int(xs[0]), int(ys[0]))))
    return first_end, farthest_pixel(geodesic_distance(mask, first_end))


def trace_path(mask, start, spacing=20):
    """Points along the line from start to its far end, sampled every spacing px."""
    distance = geodesic_distance(mask, start)
    ys, xs = np.nonzero(distance >= 0)
    # The middle of each band of equal distance follows the center of the line.
    bands = distance[ys, xs] // spacing
    path = [
        (int(xs[bands == band].mean()), int(ys[bands == band].mean()))
        for band in np.unique(bands)
    ]
    return path if len(path) > 1 else [start, farthest_pixel(distance)]


class Line:
    def __init__(self, mask, path, start_source):
        self.mask = mask
        self.contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Points from where the line started to where it ended.
        self.path = path
        # How the start was decided: "seen", "reveal order" or "arm direction".
        self.start_source = start_source


class LineDetector:
    """Find the ink added each time the arm enters and leaves the canvas."""

    def __init__(self, settle_frames=8, min_line_area=60, min_start_area=15, max_start_area=600,
                 max_start_distance=60, end_length=40, min_arm_motion=40):
        self.settle_frames = settle_frames
        self.min_line_area = min_line_area
        self.min_start_area = min_start_area
        self.max_start_area = max_start_area
        self.max_start_distance = max_start_distance
        self.end_length = end_length
        self.min_arm_motion = min_arm_motion
        self.lines = []
        self.known_ink = None
        self.state = "READY"
        self.settle_masks = []

    def _begin_stroke(self):
        # The frame number at which each pixel of new ink was first visible (-1 if not yet seen).
        self.first_seen = np.full(self.known_ink.shape, -1, np.int32)
        self.frame_index = 0
        self.start = None
        self.start_decided = False
        self.start_candidate = None
        self.arm_tips = []

    def update(self, hsv, arm_contour, arm_tip):
        """Return a new Line when one has just been detected, otherwise None."""
        ink = get_ink_mask(hsv)
        if self.known_ink is None:
            # Ink already on the canvas at startup isn't a new line.
            if arm_contour is None:
                self.known_ink = ink
            return None

        if arm_contour is not None:
            # A single frame where the arm drops out doesn't end the stroke.
            if self.state == "READY":
                self._begin_stroke()
            self.state = "DRAWING"
            self.settle_masks.clear()
            self.arm_tips.append(arm_tip)
            self._record_visible_ink(ink, arm_contour, arm_tip)
            return None
        if self.state == "READY":
            return None

        # Wait a few arm-free frames so a flickering arm or camera noise isn't mistaken for ink.
        self.state = "PROCESSING"
        self._record_visible_ink(ink, None, None)
        self.settle_masks.append(ink)
        if len(self.settle_masks) < self.settle_frames:
            return None
        line = self._extract_new_line()
        self.settle_masks.clear()
        self.state = "READY"
        return line

    def _new_ink(self, ink):
        # Grow the known ink a little so slight camera jitter doesn't count as new ink.
        known = cv2.dilate(self.known_ink, np.ones((5, 5), np.uint8), iterations=2)
        new_ink = cv2.bitwise_and(ink, cv2.bitwise_not(known))
        return cv2.morphologyEx(new_ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _record_visible_ink(self, ink, arm_contour, arm_tip):
        visible = self._new_ink(ink)
        arm = np.zeros_like(ink)
        if arm_contour is not None:
            # Ignore ink right at the arm's edge, where the arm's outline can blend into blue.
            cv2.drawContours(arm, [arm_contour], -1, 255, -1)
            arm = cv2.dilate(arm, np.ones((9, 9), np.uint8))
            visible = cv2.bitwise_and(visible, cv2.bitwise_not(arm))
        fresh = (visible > 0) & (self.first_seen < 0)
        self.first_seen[fresh] = self.frame_index
        if not self.start_decided:
            self._check_start(visible > 0, arm > 0, fresh, arm_tip)
        self.frame_index += 1

    def _check_start(self, visible, hidden, fresh, arm_tip):
        if self.start_candidate is not None:
            candidate, candidate_frame, candidate_tip = self.start_candidate
            self.start_candidate = None
            # Trust the first ink only if it is still there a frame later, otherwise it was noise.
            # Ink the arm has moved over can't be checked, so it counts as still there.
            in_view = candidate & ~hidden
            if np.count_nonzero(visible[in_view]) * 2 >= np.count_nonzero(in_view):
                self.start_decided = True
                self.start = self._certain_start(candidate, candidate_tip)
                return
            self.first_seen[candidate & (self.first_seen == candidate_frame)] = -1
        if np.count_nonzero(fresh) >= self.min_start_area:
            self.start_candidate = (fresh, self.frame_index, arm_tip)

    def _certain_start(self, candidate, arm_tip):
        # The first ink is the start only if it is a small dot right at the pen. A long piece, or ink
        # uncovered away from the pen, means the start was hidden under the arm.
        if arm_tip is None or np.count_nonzero(candidate) > self.max_start_area:
            return None
        ys, xs = np.nonzero(candidate)
        if np.min(np.hypot(xs - arm_tip[0], ys - arm_tip[1])) > self.max_start_distance:
            return None
        return int(xs.mean()), int(ys.mean())

    def _extract_new_line(self):
        votes = np.sum([mask > 0 for mask in self.settle_masks], axis=0)
        stable_ink = np.where(votes * 2 >= len(self.settle_masks), 255, 0).astype(np.uint8)
        new_ink = self._new_ink(stable_ink)

        contours, _ = cv2.findContours(new_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_line_area]
        if not contours:
            return None
        keep = np.zeros_like(new_ink)
        cv2.drawContours(keep, contours, -1, 255, -1)
        mask = cv2.bitwise_and(new_ink, keep)
        # Trace only the biggest piece; the others are still kept in the mask.
        main_piece = np.zeros_like(mask)
        cv2.drawContours(main_piece, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        main_piece = cv2.bitwise_and(mask, main_piece)

        start, start_source = self._choose_start(main_piece)
        line = Line(mask, trace_path(main_piece, start), start_source)
        self.lines.append(line)
        self.known_ink = cv2.bitwise_or(self.known_ink, line.mask)
        return line

    def _choose_start(self, mask):
        end_a, end_b = find_endpoints(mask)
        if self.start is not None:
            return nearest_end(self.start, end_a, end_b), "seen"

        # The end that became visible first is where the line started.
        reveal_a = self._reveal_time_near(mask, end_a)
        reveal_b = self._reveal_time_near(mask, end_b)
        if reveal_a != reveal_b:
            return (end_a if reveal_a < reveal_b else end_b), "reveal order"

        # Nothing was visible while drawing: the line started at the end the arm moved away from.
        tips = [tip for tip in self.arm_tips if tip is not None]
        if len(tips) >= 2:
            motion = np.subtract(tips[-1], tips[0])
            if np.hypot(*motion) >= self.min_arm_motion:
                along_b = np.dot(np.subtract(end_b, end_a), motion)
                return (end_a if along_b > 0 else end_b), "arm direction"
        return end_a, "guess"

    def _reveal_time_near(self, mask, end):
        near = (geodesic_distance(mask, end) >= 0) & (geodesic_distance(mask, end) <= self.end_length)
        times = self.first_seen[near]
        times = times[times >= 0]
        return float(np.median(times)) if len(times) else float("inf")


def nearest_end(point, end_a, end_b):
    distance_a = np.hypot(end_a[0] - point[0], end_a[1] - point[1])
    distance_b = np.hypot(end_b[0] - point[0], end_b[1] - point[1])
    return end_a if distance_a <= distance_b else end_b


DRAWING_STATE_COLORS = {"DRAWING": (0, 160, 0), "PROCESSING": (0, 140, 255), "READY": (80, 80, 80)}


def draw_start_marker(frame, point):
    x, y = point
    cv2.circle(frame, point, 14, (0, 0, 255), 3)
    cv2.line(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 3)
    cv2.line(frame, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 3)


def draw_path(frame, path, color):
    for begin, end in zip(path, path[1:]):
        cv2.arrowedLine(frame, begin, end, color, 2, tipLength=0.5)


def draw_lines(frame, line_detector):
    for line in line_detector.lines[:-1]:
        draw_path(frame, line.path, (0, 140, 140))
    drawing = line_detector.state != "READY"
    if line_detector.lines:
        latest = line_detector.lines[-1]
        draw_path(frame, latest.path, (0, 255, 255))
        if not drawing:
            draw_start_marker(frame, latest.path[0])
    if drawing and line_detector.start is not None:
        draw_start_marker(frame, line_detector.start)
    state = line_detector.state
    label = f"{state} - lines: {len(line_detector.lines)}"
    if line_detector.lines and not drawing:
        label += f" (start: {line_detector.lines[-1].start_source})"
    cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, DRAWING_STATE_COLORS[state], 2)


def should_quit():
    # Drain repeated key presses so window updates don't get held up.
    while (key := cv2.pollKey()) != -1:
        if key & 0xFF in (ord("q"), ord("Q")):
            return True
    return False


async def show_camera(websocket):
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    camera = cv2.VideoCapture(0, backend)
    window = "Camera - Q to quit"
    arm_tracker = ArmDirectionTracker()
    line_detector = LineDetector()
    calibrator = CanvasCalibrator()
    canvas_transform = None
    try:
        if not camera.isOpened():
            print("Could not open camera")
            return

        while websocket.close_code is None:
            ok, frame = camera.read()
            if not ok:
                break

            if canvas_transform is None:
                corners = calibrator.update(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
                if corners is not None:
                    canvas_transform, canvas_size = get_canvas_transform(corners)
                    print(f"Canvas corners: {corners.astype(int).tolist()}")
                else:
                    draw_calibration(frame, calibrator.samples[-1] if calibrator.samples else None)
                    cv2.imshow(window, frame)
                    if should_quit():
                        return
                    await asyncio.sleep(0.01)
                    continue

            # Everything below works on a top-down view of just the canvas.
            frame = cv2.warpPerspective(frame, canvas_transform, canvas_size)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            arm_contour, arm_tip = get_arm(hsv)
            arm_direction, arm_motion = arm_tracker.update(arm_tip)
            new_line = line_detector.update(hsv, arm_contour, arm_tip)
            if new_line is not None:
                print(f"New line #{len(line_detector.lines)}: {cv2.countNonZero(new_line.mask)} px, "
                      f"start {new_line.path[0]} ({new_line.start_source})")

                width, height = canvas_size
                await websocket.send(json.dumps({
                    "type": "LINE",
                    "path": new_line.path,
                    "width": width,
                    "height": height,
                }))

            draw_arm(frame, arm_contour, arm_tip, arm_direction, arm_motion)
            draw_lines(frame, line_detector)
            cv2.imshow(window, frame)
            if should_quit():
                return

            # Let the server process connection events between frames.
            await asyncio.sleep(0.01)
    except ConnectionClosed:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed")


async def handle_connection(websocket):
    print("Client connected")
    await show_camera(websocket)


async def main():
    async with serve(handle_connection, "localhost", 8765) as server:
        print("Waiting for connections on ws://localhost:8765")
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
