# PenPal

## Setup
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

## To run the UI

streamlit run UI/canvas.py

## Plotter

The plotter code is in `plotter/`. It needs `pip install -r UI/requirements.txt` (matplotlib, pyserial).
Machine settings (paper position, pen up/down commands, serial port) are at the top of `plotter/gcode.py`.

In the app, a plotter preview appears under the canvas once the model stops drawing, with buttons to download the G-code or send it to the plotter.

```
python plotter/make_samples.py                                          # test drawings in plotter/output
python plotter/simulate.py plotter/output/sample_ruler.gcode --animate  # watch it draw, no hardware
python plotter/send.py plotter/output/sample_ruler.gcode --dry-run      # practice send, no hardware
python plotter/send.py plotter/output/sample_ruler.gcode --check        # Arduino checks every line, no movement
python plotter/send.py plotter/output/sample_ruler.gcode                # plot it
```

The Arduino runs GRBL with servo support. It treats wherever the pen is when a program connects as X0 Y0, so put the pen at the home corner before sending. Every job ends back there.
