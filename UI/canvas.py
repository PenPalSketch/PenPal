import streamlit as st
import random
from streamlit_drawable_canvas import st_canvas


AVAILABLE_MODELS = ['bird', 'ant','ambulance','angel','alarm_clock','antyoga','backpack','barn','basket','bear','bee','beeflower','bicycle','book','brain','bridge','bulldozer','bus','butterfly','cactus','calendar','castle','cat','catbus','catpig','chair','couch','crab','crabchair','crabrabbitfacepig','cruise_ship','diving_board','dog','dogbunny','dolphin','duck','elephant','elephantpig','everything','eye','face','fan','fire_hydrant','firetruck','flamingo','flower','floweryoga','frog','frogsofa','garden','hand','hedgeberry','hedgehog','helicopter','kangaroo','key','lantern','lighthouse','lion','lionsheep','lobster','map','mermaid','monapassport','monkey','mosquito','octopus','owl','paintbrush','palm_tree','parrot','passport','peas','penguin','pig','pigsheep','pineapple','pool','postcard','power_outlet','rabbit','rabbitturtle','radio','radioface','rain','rhinoceros','rifle','roller_coaster','sandwich','scorpion','sea_turtle','sheep','skull','snail','snowflake','speedboat','spider','squirrel','steak','stove','strawberry','swan','swing_set','the_mona_lisa','tiger','toothbrush','toothpaste','tractor','trombone','truck','whale','windmill','yoga','yogabicycle'];

ss = st.session_state
ss.setdefault("canvas_key", 0)
if 'model' not in ss:
    ss.model = AVAILABLE_MODELS[0]

def clear():
    ss.canvas_key += 1 # new key results in a new empty canvas
    ss.all_raw_lines = []
    ss.strokes = []
    ss.object_count = 0
    
def pick_random_model():
    ss.model = random.choice(AVAILABLE_MODELS)

# this variable contains lines in form of lists of points: list([x, y]), !! not deltas
ss.setdefault("all_raw_lines", [])
# contains all the strokes
ss.setdefault("strokes", [])
ss.setdefault("object_count", 0)

st.title("Pen Pal Canvas")
st.subheader("Interactive Sketch Prediction")
st.write("This demo attempts to finish the drawing given whatever strokes you draw on the screen. You can also select other classes, like 'cat', 'ant', 'bus', etc.")

col1, col2, col3 = st.columns(3, vertical_alignment="bottom")

with col1:
    st.button("Clear", on_click=clear, use_container_width=True)
with col2:
    st.button("Random", on_click=pick_random_model, use_container_width=True)
with col3:
    st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        key="model",
    )

def on_temp_change():
    print(f"Temperature changed: {ss.temperature}")

temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    value=1.0,
    key="temperature",
    on_change=on_temp_change)

canvas_result = st_canvas(
    drawing_mode="freedraw",
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_width=2,
    stroke_color="#000000",
    background_color="#EEEEEE",
    update_streamlit=True,
    width=700,
    height=400,
    return_image_data=True,
    key=f"canvas{ss.canvas_key}",
)


# converts the path of the stroke into the (x, y) format.
#
# streamlit returns a path of a line in an SVG path format,
# so these aren't exact mouse positions, but approximations
def get_points(path):
    points = []
    for c in path:
        if c[0] == 'M':
            points.append([c[1], -c[2]])
        elif c[0] == 'Q':
            x, y = c[3], -c[4]
            points.append([x, y])
        elif c[0] == 'L':
            x, y = c[1], -c[2]
            points.append([x, y])

    return points

def on_new_line(raw_lines):
    if not len(raw_lines) > 0: 
        return

    raw_line_svg = raw_lines[-1]

    # path of the entire line user drew
    path = raw_line_svg["path"]

    # list of raw points converted from streamlit's SVG path
    # raw_line_points is equivalent to currentRawLine in mouseReleased from JS
    raw_line = get_points(path)

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
    if len(objects) > ss.object_count:
        on_new_line(objects)
        ss.object_count = len(objects)
    
    
