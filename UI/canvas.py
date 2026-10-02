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
    
def pick_random_model():
    ss.model = random.choice(AVAILABLE_MODELS)

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

temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.01, value=1.0)

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