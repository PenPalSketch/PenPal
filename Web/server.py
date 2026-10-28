import asyncio
from ctypes import windll

import cv2
from websockets.asyncio.server import serve


def is_spacebar_down():
    return bool(windll.user32.GetAsyncKeyState(0x20) & 0x8000)


def get_pen_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (45, 60, 40), (85, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea, default=None)
    if contour is None or cv2.contourArea(contour) < 50:
        return None

    x, y, width, height = cv2.boundingRect(contour)
    return x + width // 2, y + height // 2


async def show_camera(websocket):
    camera = cv2.VideoCapture(0)
    window = "Camera - Q to quit"
    try:
        if not camera.isOpened():
            print("Could not open camera")
            return

        while websocket.close_code is None:
            ok, frame = camera.read()
            if not ok:
                break

            center = get_pen_center(frame)
            if center is not None:
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

            pen_down = is_spacebar_down()
            status = "PEN DOWN" if pen_down else "PEN UP"
            color = (0, 200, 0) if pen_down else (0, 0, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow(window, frame)
            # Drain repeated key presses so window updates don't get held up.
            while (key := cv2.pollKey()) != -1:
                if key & 0xFF in (ord("q"), ord("Q")):
                    return

            # Let the server process connection events between frames.
            await asyncio.sleep(0.01)
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
