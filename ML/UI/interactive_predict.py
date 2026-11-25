import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.model import SketchRNN

ss = st.session_state
MODEL_STROKES_PER_TURN = 3
MAX_MODEL_STEPS = 500

# same as js's initModel()
def load_model():
    model = SketchRNN("url")
    model.initialize()
    model.setPixelFactor(5.0)   # bigger -> larger drawings
    return model

def predict_turn(update_canvas=True):
    """Generate a reply and add its drawn moves to the shared sketch."""
    model = ss.model
    state = model.update(model.zero_input(), model.zeroState())
    state = model.updateStrokes(ss.strokes, state, len(ss.strokes) - 1)
    previous = ss.strokes[-1]
    ss.model_anchor = (ss.x, ss.y)

    moves = []
    last_drawn_move = 0
    completed_strokes = 0
    drew_current_stroke = False
    x, y = ss.model_anchor
    for _ in range(MAX_MODEL_STEPS):
        state = model.update(previous, state)
        dx, dy, pen_down, pen_up, pen_end = model.sample(model.getPDF(state, ss.temperature))
        if pen_end:
            break
        if update_canvas:
            dx, dy = int(dx), int(dy)

        _, _, previous_pen_down, _, _ = previous
        previous = [dx, dy, pen_down, pen_up, pen_end]
        moves.append(previous)
        if previous_pen_down and (dx or dy):
            if update_canvas:
                draw_line(x, y, x + dx, y + dy)
            last_drawn_move = len(moves)
            drew_current_stroke = True
        x += dx
        y += dy

        if pen_up and drew_current_stroke:
            completed_strokes += 1
            drew_current_stroke = False
            if completed_strokes == MODEL_STROKES_PER_TURN:
                break

    # Keep travel leading into visible ink. End at the last visible point with the pen up.
    ss.model_strokes = moves[:last_drawn_move]
    if ss.model_strokes:
        ss.model_strokes[-1][2:] = [0, 1, 0]
        ss.strokes.extend(ss.model_strokes)
        ss.x = ss.model_anchor[0] + sum(move[0] for move in ss.model_strokes)
        ss.y = ss.model_anchor[1] + sum(move[1] for move in ss.model_strokes)


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
