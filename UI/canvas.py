import sys
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import random
from streamlit_drawable_canvas import st_canvas
from UI.interactive_predict import AVAILABLE_MODELS

CANVAS_WIDTH = 700
CANVAS_HEIGHT = 400

# State (equivalent to the "let" variables from JS)
ss = st.session_state
ss.setdefault("canvas_key", 0)                    # increase to get a fresh, empty canvas
ss.setdefault("model_name", AVAILABLE_MODELS[0]) 
ss.setdefault("temperature", 0.25)
ss.setdefault("all_raw_lines", [])                 # contains lines in form of lists of points: list([x, y]), !! not deltas
ss.setdefault("strokes", [])                       # contains all the strokes [dx, dy, p1, p2, p3]]
ss.setdefault("object_count", 0)                   # canvas objs already converted to strokes
ss.setdefault("model_state", None)

# Update Functions
def restart():
    ss.canvas_key += 1 # new key results in a new empty canvas
    ss.all_raw_lines = []
    ss.strokes = []
    ss.object_count = 0
    ss.model_state = None
    
def pick_random_model():
    ss.model_name = random.choice(AVAILABLE_MODELS)

# Layout (UI and DOM elements)
st.title("Pen Pal Canvas")
st.subheader("Interactive Sketch Prediction")
st.write("This demo attempts to finish the drawing given whatever strokes you draw on the screen. You can also select other classes, like 'cat', 'ant', 'bus', etc.")

col1, col2, col3 = st.columns(3, vertical_alignment="bottom")

with col1:
    st.button("Clear", on_click=restart, use_container_width=True)
with col2:
    st.button("Random", on_click=pick_random_model, use_container_width=True)
with col3:
    st.selectbox("Model", AVAILABLE_MODELS, key="model_name")

st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    key="temperature")

canvas_result = st_canvas(
    drawing_mode="freedraw",
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_width=2,
    stroke_color="#FF0000", # JS says that the user always draws in red
    background_color="#EEEEEE",
    update_streamlit=True,
    width=CANVAS_WIDTH,
    height=CANVAS_HEIGHT,
    return_image_data=True,
    key=f"canvas{ss.canvas_key}",
)


# equivalent to mouseDragged
# Note: Origin (0,0) sits in the top left corner
# Converts a fabric.js path (of the stroke) into the (x, y) format.
# Returns a list of (x, y) points
# streamlit returns a path of a line in an SVG path format,
# so these aren't exact mouse positions, but approximations
def get_points(path):
    points = []
    for c in path:
        if c[0] == 'M':                   # ["M", x, y] - Move to point
            points.append([c[1], c[2]])
        elif c[0] == 'Q':                 # ["Q", control_x, control_y, end_x, end_y], - Curve
            x, y = c[3], c[4]
            points.append([x, y])
        elif c[0] == 'L':                 # ["L", next_x, next_y]  - Line
            x, y = c[1], c[2]
            points.append([x, y])
    return points


# equivalent to mouseReleased
def on_new_line(raw_lines):
    if not len(raw_lines) > 0: # nothing drawn
        return

    raw_line_svg = raw_lines[-1] # newest stroke

    # path of the entire line user drew
    path = raw_line_svg["path"] # its [["M", x, y], ["Q", ...], ...], etc.

    # list of raw points converted from streamlit's SVG path
    # raw_line_points is equivalent to currentRawLine in mouseReleased from JS
    raw_line = get_points(path) # [[x, y], [x, y], ...]

    ss.all_raw_lines.append(raw_line)


    raw_line_simplified = []
    # raw_line_simplified = model.simplifyLine(raw_line_points)

    # the end point of previous line is needed
    # see p.mouseReleased in JS
    #   this will make the first displacement of this line very big
    #   which is useful information because model knows:
    #   the human ended this line here and then started a new line all the way here
    #   and that is somehow useful

    prev_line_end_point = (0, 0);
    if len(ss.all_raw_lines) > 1:
        prev_line_end_point = ss.all_raw_lines[-2][-1]

    # stroke = model.lineToStroke(raw_line_simplified, prev_line_end_point)

    # strokes = ss.strokes.concat(stroke)
    # initRNNStateFromStrakes(strokes)



# this is called whenever a new stroke was drawn
if canvas_result.json_data is not None:
    print("new line")

    objects = canvas_result.json_data.get("objects", [])
    if len(objects) > ss.object_count:                    # more strokes than previous drawing
        on_new_line(objects)
        ss.object_count = len(objects)                    # save # of objs seen
