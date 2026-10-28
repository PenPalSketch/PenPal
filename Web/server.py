import asyncio

import cv2
from websockets.asyncio.server import serve


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

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

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
