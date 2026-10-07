import sys
from pathlib import Path
import random

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pen index constants
PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2

ss = st.session_state

AVAILABLE_MODELS = ['bird', 'ant','ambulance','angel','alarm_clock','antyoga','backpack','barn','basket','bear','bee','beeflower','bicycle','book','brain','bridge','bulldozer','bus','butterfly','cactus','calendar','castle','cat','catbus','catpig','chair','couch','crab','crabchair','crabrabbitfacepig','cruise_ship','diving_board','dog','dogbunny','dolphin','duck','elephant','elephantpig','everything','eye','face','fan','fire_hydrant','firetruck','flamingo','flower','floweryoga','frog','frogsofa','garden','hand','hedgeberry','hedgehog','helicopter','kangaroo','key','lantern','lighthouse','lion','lionsheep','lobster','map','mermaid','monapassport','monkey','mosquito','octopus','owl','paintbrush','palm_tree','parrot','passport','peas','penguin','pig','pigsheep','pineapple','pool','postcard','power_outlet','rabbit','rabbitturtle','radio','radioface','rain','rhinoceros','rifle','roller_coaster','sandwich','scorpion','sea_turtle','sheep','skull','snail','snowflake','speedboat','spider','squirrel','steak','stove','strawberry','swan','swing_set','the_mona_lisa','tiger','toothbrush','toothpaste','tractor','trombone','truck','whale','windmill','yoga','yogabicycle']
BASE_URL = "https://storage.googleapis.com/quickdraw-models/sketchRNN/models/"

from ML.model import SketchRNN

# same as js's initModel()
def load_model():
    model = SketchRNN("url")
    model.initialize()
    print("SketchRNN model loaded")
    model.setPixelFactor(5.0)   # bigger -> larger drawings
    return model

# Feeds user's strokes through the RNN so it remembers the drawing
# Return model state or None if there aren't enough strokes yet 
def encode_strokes(model, strokes):
    if len(strokes) <= 5:
        return None

    new_state = model.zeroState()
    new_state = model.update(model.zeroInput(), new_state)
    new_state = model.updateStrokes(strokes, new_state, len(strokes) - 1)
    return model.copyState(new_state)


def initRNNStateFromStrokes(strokes):
    # Initialize the RNN with these strokes.
    encode_strokes(strokes);

    # JS redraws user strokes for some reason, i'll comment it out for now
    # Draw them.
    # draw_strokes(strokes, startX, startY);



# p.draw in JS
def draw(model):
    prev_pen = ss.prev_pen_state
    model_state = model.update([ss.dx, ss.dy] + prev_pen, ss.model_state)
    pdf = model.getPDF(model_state, ss.temperature)

    sample = model.sample(pdf)

    dx = sample[0]
    dy = sample[1]
    pen_state = sample[2:]

    # if previous drawing is finished, start a new one
    if pen_state[PEN_END] == 1:
        # initRNNStateFromStrokes(ss.strokes)
        pass
    else:
        if prev_pen[PEN_DOWN] == 1:
            # draw line
            # p.line
            pass
        # update
        ss.x += dx
        ss.y += dy
        ss.prev_pen_state = pen_state
