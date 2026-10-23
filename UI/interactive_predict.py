import sys
from pathlib import Path
import random
import math
import streamlit as st
from ML.model import SketchRNN

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pen index constants
PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2

ss = st.session_state

# same as js's initModel()
def load_model():
    model = SketchRNN("url")
    model.initialize()
    model.setPixelFactor(5.0)   # bigger -> larger drawings
    return model

# Feeds user's strokes through the RNN so it remembers the drawing
# Return model state or None if there aren't enough strokes yet 
def encode_strokes(strokes):
    model = ss.model

    if len(strokes) <= 5:
        return None


    new_state = model.zeroState()
    new_state = model.update(model.zero_input(), new_state)
    new_state = model.updateStrokes(strokes, new_state, len(strokes) - 1)

    ss.model_state = model.copyState(new_state)

    # set x and y
    last_point = ss.all_raw_lines[-1][-1]
    last_last_point = ss.all_raw_lines[-1][-2]

    ss.x = last_point[0]
    ss.y = last_point[1]

    ss.dx = last_last_point[0] - last_point[0]
    ss.dy = last_last_point[1] - last_point[1]



def initRNNStateFromStrokes(strokes):
    encode_strokes(strokes)
    ss.model_drawing = True
    draw()

    # JS redraws user strokes for some reason, i'll comment it out for now
    # Draw them.
    # draw_strokes(strokes, startX, startY);



def draw_line(x1, y1, x2, y2, color="#000000", width=2):
    left, top = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)

    cx, cy = left + w / 2, top + h / 2

    # add object count so that this line isn't treated as a line
    # a user drew, so it wouldnt get converted to a stroke
    ss.generated_lines.append({
        "type": "line",
        "originX": "left", "originY": "top",
        "left": left, "top": top,
        "width": w, "height": h,
        "x1": x1 - cx, "y1": y1 - cy,
        "x2": x2 - cx, "y2": y2 - cy,
        "stroke": color,
        "strokeWidth": width,
        "strokeLineCap": "round",
    })
    ss.object_count += 1

# p.draw in JS
def draw():
    ss.model_strokes_count += 1

    # limit how many strokes the model draws
    # for now because it never stops
    if ss.model_strokes_count > 500:
        return

    model = ss.model
    prev_pen = ss.prev_pen_state
    ss.model_state = model.update([ss.dx, ss.dy] + prev_pen, ss.model_state)
    pdf = model.getPDF(ss.model_state, ss.temperature)

    sample = model.sample(pdf)

    dx = int(sample[0])
    dy = int(sample[1])
    pen_state = sample[2:]

    # if pen_end = 1, end
    if pen_state[PEN_END] == 1:
        ss.model_drawing = False
        return

    # if previous drawing is finished, start a new one
    if pen_state[PEN_END] == 1:
        # initRNNStateFromStrokes(ss.strokes)
        pass
    else:
        if prev_pen[PEN_DOWN] == 1:
            draw_line(ss.x, ss.y, ss.x + dx, ss.y + dy)
            pass
        # update
        ss.x += dx
        ss.y += dy
        ss.dx = dx
        ss.dy = dy
        ss.prev_pen_state = pen_state

        st.rerun()


