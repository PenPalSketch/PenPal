import asyncio

from websockets.asyncio.server import serve


async def handle_connection(websocket):
    print("Client connected")
    await websocket.wait_closed()
    print("Client disconnected")


async def main():
    async with serve(handle_connection, "localhost", 8765) as server:
        print("Waiting for connections on ws://localhost:8765")
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
