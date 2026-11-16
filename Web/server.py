import asyncio
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from CV.board import BoardCalibration
from HW.Plotter.gcode import CANVAS_WIDTH, CANVAS_HEIGHT

if platform.system() == "Windows":
    from ctypes import windll

    def is_spacebar_down():
        return bool(windll.user32.GetAsyncKeyState(0x20) & 0x8000)
else:
    from pynput import keyboard as _kb

    _keys_down: set = set()

    def _on_press(key):
        _keys_down.add(key)

    def _on_release(key):
        _keys_down.discard(key)

    _kb.Listener(on_press=_on_press, on_release=_on_release, daemon=True).start()

    def is_spacebar_down():
        return _kb.Key.space in _keys_down


def get_pen_center(hsv):
    mask = cv2.inRange(hsv, (45, 60, 40), (85, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea, default=None)
    if contour is None or cv2.contourArea(contour) < 50:
        return None

    x, y, width, height = cv2.boundingRect(contour)
    return x + width // 2, y + height // 2


def is_led_pen_down(hsv, threshold=300):
    # Red wraps around in HSV — check both ends of the hue range.
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 210, 210), (10, 255, 255)),
        cv2.inRange(hsv, (170, 220, 220), (180, 255, 255)),
    )
    return cv2.countNonZero(mask) > threshold


def draw_pen_status(frame, pen_down):
    status = "PEN DOWN" if pen_down else "PEN UP"
    color = (0, 200, 0) if pen_down else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


async def show_camera(websocket):
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    camera = cv2.VideoCapture(0, backend)
    window = "Camera - Q to quit"
    drawing = False
    previous_pen_down = False
    stroke_recorded = False
    calibration = BoardCalibration(CANVAS_WIDTH, CANVAS_HEIGHT)
    calibrated = False

    def select_corner(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration.add_corner(x, y)

    try:
        if not camera.isOpened():
            print("Could not open camera")
            return

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, select_corner)
        await websocket.send(json.dumps({"type": "CALIBRATING"}))
        while websocket.close_code is None:
            ok, frame = camera.read()
            if not ok:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            center = get_pen_center(hsv)
            if center is not None:
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

            if calibration.transform is not None and not calibrated:
                calibrated = True
                await websocket.send(json.dumps({
                    "type": "READY", "width": CANVAS_WIDTH, "height": CANVAS_HEIGHT,
                }))
            point = calibration.point(center)
            pen_down = calibrated and (is_spacebar_down() or is_led_pen_down(hsv))
            if previous_pen_down and not pen_down and stroke_recorded:
                await websocket.send(json.dumps({"type": "COMPLETE"}))
                drawing = False
                stroke_recorded = False
            elif pen_down and point is not None:
                await websocket.send(json.dumps({
                    "pen_down": True,
                    "x": point[0], "y": point[1],
                    "width": CANVAS_WIDTH, "height": CANVAS_HEIGHT,
                }))
                drawing = True
                stroke_recorded = True
            elif drawing:
                await websocket.send(json.dumps({"pen_down": False}))
                drawing = False
            previous_pen_down = pen_down

            draw_pen_status(frame, pen_down)
            calibration.draw(frame)
            cv2.imshow(window, frame)
            # Drain repeated key presses so window updates don't get held up.
            while (key := cv2.pollKey()) != -1:
                if key & 0xFF in (ord("q"), ord("Q")):
                    return
                if key & 0xFF in (ord("c"), ord("C")):
                    calibration.reset()
                    calibrated = drawing = previous_pen_down = stroke_recorded = False
                    await websocket.send(json.dumps({"type": "CALIBRATING"}))

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
