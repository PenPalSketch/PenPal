in terminal one, run this:

```
.\.venv\Scripts\python.exe Web/server.py
```

in terminal two, run this:

```
.\ML\.venv311\Scripts\python.exe -m Web.inference
```

in terminal three, go to `Web` and run

```
serve
```

Open localhost:3000 and click Connect. In the camera preview, click the drawing
area's top-left, top-right, bottom-right, and bottom-left corners, in that order.
The board is 14 inches wide by 12 inches high (355.6 × 304.8 mm), represented by a
700 × 600 canvas. Calibration corrects the camera perspective and excludes the
space outside the board. Press C in the preview to select the corners again if
the camera or board moves. Reconnecting also requires selecting the corners again.

Hold Space while moving the green pen, then release it for a model continuation.
The continuation is centred inside the largest closed human outline and shrunk
uniformly when needed, with clearance from its edge. For open drawings it uses
their bounding rectangle. The browser and G-code use the same fitted strokes.

Machine dimensions, axis directions, and fixed offsets are in `HW/Plotter/gcode.py`.
Generated drawing coordinates subtract 80 mm from X and 75 mm from Y after
canvas scaling and axis flips. The simulator uses the translated drawing bounds.
Each job sets its starting pen position to X0 Y0 and returns there at the end.
The firmware must be calibrated to move in millimetres.
