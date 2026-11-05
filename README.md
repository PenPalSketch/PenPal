# PenPal ✏️

## Setup
python -m venv .venv

source .venv/bin/activate

pip install -r ML/requirements.txt

pip install -r CV/requirements.txt

pip install -r HW/requirements.txt

## To run the UI

streamlit run ML/UI/canvas.py

## Twenty-turn model self-play demo

From the repository root, run:

```powershell
.\ML\.venv311\Scripts\python.exe -m ML.self_play
```

Open `ML/output/self_play.gif` to watch twenty strokes appear in order, with a pause
and a new color for each turn. The animation loops; generation happens once.
`ML/output/self_play.json` contains each turn's points and raw stroke-5 samples.
No camera, server, plotter, or human seed stroke is needed. The model starts from
its zero state and start token, then feeds every generated movement back into
the same recurrent state. One turn is a continuous pen-down stroke ending at a
pen lift; pen-up travel is retained in model state but is not drawn or counted.

The demo disables the end-of-drawing token to reach exactly twenty strokes. This
constrains sampling and may add strokes after the model would naturally finish.
The existing inference and UI behavior are unchanged. Generation raises an error
if any turn exceeds 500 sampling steps instead of counting an unfinished stroke.

The script has no CLI options. Edit `NUM_TURNS` and `TEMPERATURE` at the top of
`ML/self_play.py` to change the demo. It uses a fixed random seed and the existing
TensorFlow and Pillow dependencies.

## Plotter
Machine settings (paper position, pen up/down commands, serial port) are at the top of `Plotter/gcode.py`.

In the app, a plotter preview appears under the canvas once the model stops drawing, with buttons to download the G-code or send it to the plotter.

```
python Plotter/make_samples.py                                          # test drawings in plotter/output
python Plotter/simulate.py plotter/output/sample_ruler.gcode --animate  # watch it draw, no hardware
python Plotter/send.py plotter/output/sample_ruler.gcode --dry-run      # practice send, no hardware
python Plotter/send.py plotter/output/sample_ruler.gcode --check        # Arduino checks every line, no movement
python Plotter/send.py plotter/output/sample_ruler.gcode                # plot it
```

The Arduino runs GRBL with servo support. It treats wherever the pen is when a program connects as X0 Y0, so put the pen at the home corner before sending. Every job ends back there.
