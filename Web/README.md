# Camera drawing and completion

Run these commands from the repository root in two separate terminals:

```powershell
.\.venv\Scripts\python.exe Web/server.py
```

```powershell
.\ML\.venv311\Scripts\python.exe -m Web.inference
```

Open **http://127.0.0.1:8000** (rather than opening the HTML file directly).
You can also serve the frontend on **http://localhost:3000**; completion requests
always go to **http://localhost:8000/complete**.
Click **Connect**, hold Space while moving the green pen, and release it to end
a stroke and automatically generate a cat-drawing continuation in blue.
Each release submits the entire drawing history, including earlier model strokes.
The response is drawn directly onto the canvas and appended to that history.
Completion requests run one at a time. Losing sight of the pen breaks the drawn
line but doesn't trigger inference; the camera server sends `{"type":"COMPLETE"}`
only when Space is released.
**Clear** erases the drawing and history. Q closes the camera session;
Ctrl+C stops a server.

The inference server uses the existing Python 3.11 environment. For a fresh
Python 3.11 environment, install `Web/requirements-inference.txt`; the camera
server's dependencies are in `Web/requirements.txt`.

## Endpoint

`POST /complete` accepts absolute pixel coordinates grouped into separate strokes:

```json
{"strokes": [[[100, 100], [110, 90], [120, 100]], [[140, 100], [150, 110]]], "temperature": 0.25}
```

It returns `{"strokes": [...], "finished": true}`. These are only the generated
paths, in the same coordinate system as the input. `finished` is false if
generation reaches the 500-step limit. Temperature is optional (default 0.25,
range 0–1). Each request starts with fresh drawing state; the cat model stays
loaded. The caller supplies valid numeric coordinates and temperature values.
Requests with no usable stroke return HTTP 400.
