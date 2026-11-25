# Camera drawing

Install the dependencies from `CV/requirements.txt` and `ML/requirements.txt`.
From the repository root, run each command in its own terminal:

```sh
python CV/server.py
streamlit run ML/UI/canvas.py
```

Open [Streamlit](http://localhost:8501), select **Camera**, and click
**Connect camera**. This opens the camera preview on the machine running the
camera server.

Before starting it, place one small blue dot at each paper corner. The dots
should be about **3 mm in diameter**, solid, and as close to the same size as
possible:

```text
blue dot             blue dot
       +-----------+
       |   paper   |
       +-----------+
blue dot             blue dot
```

The centre of each dot defines that paper corner, so place it accurately and
keep the surrounding 5 mm clear of other blue ink. Keep all four dots in view
and keep the board clear during the two-second startup calibration. The green
outline shows the detected paper area. After calibration, the perspective
transform is frozen so new blue drawing cannot change the scale. Restart the
camera connection if the camera or paper moves. The rectified image imports the
physical bounds `(0, 0, 370, 290)` from `HW/Plotter/gcode.py`, at two pixels per
millimetre.

The Streamlit **Plotter calibration** section has a **Draw corner dots** button.
It lowers the marker at each exact paper bound and returns home. Enable
**Practice run (pretend Arduino)** first to test the generated job without
moving the machine.

Draw a blue line,
then move your hand out of the camera view. The camera traces the new ink
after eight frames without a detected arm and sends the line to Streamlit
for a model continuation. Use **Clear** to reset the Streamlit drawing or
**Disconnect camera** to close the connection.

Streamlit previews the generated G-code and sends each new continuation to the
plotter once. Enable **Practice run (pretend Arduino)** before drawing to test
the workflow without moving the plotter. Select **Draw in Streamlit** to use the
mouse canvas.

## Camera-to-plotter copy test

To check camera and plotter offsets, draw a few thin blue crosses or straight
lines on the paper and use a contrasting colour in the plotter pen. Start with
a practice run:

```sh
python CV/test_copy.py --dry-run
```

Wait for calibration, then press **Space** to capture the board. The red preview
is the centreline the plotter will follow. Press **P** to run the practice copy,
**R** to capture again, or **Q** to quit. The preview and generated G-code are
saved as `HW/Plotter/output/test_copy_preview.png` and `test_copy.gcode`.

When the preview is correct, put the plotter pen at its home corner and run:

```sh
python CV/test_copy.py
```

Nothing moves until **P** is pressed on the preview. Keep a hand near the
plotter power switch during the first physical test.

The copy test uses the same plotter flow as Streamlit: traced paths are
converted to stroke-5 data, mapped through `CanvasToPaper`, passed to
`strokes_to_gcode` and `write_gcode` in `HW/Plotter/gcode.py`, then sent by
`send_gcode` in `HW/Plotter/send.py`.
