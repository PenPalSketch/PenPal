# WebSocket server

From the repository root, with your virtual environment activated:

```powershell
python -m pip install -r Web/requirements.txt
python Web/server.py
```

Connect to `ws://localhost:8765`. The server prints when a client connects or
disconnects and keeps each connection open until the client closes it.
Press Ctrl+C to stop the server.
