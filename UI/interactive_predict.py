import streamlit as st
import random
# Pen index constants
PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2

<<<<<<< HEAD
ss = st.session_state

AVAILABLE_MODELS = ['bird', 'ant','ambulance','angel','alarm_clock','antyoga','backpack','barn','basket','bear','bee','beeflower','bicycle','book','brain','bridge','bulldozer','bus','butterfly','cactus','calendar','castle','cat','catbus','catpig','chair','couch','crab','crabchair','crabrabbitfacepig','cruise_ship','diving_board','dog','dogbunny','dolphin','duck','elephant','elephantpig','everything','eye','face','fan','fire_hydrant','firetruck','flamingo','flower','floweryoga','frog','frogsofa','garden','hand','hedgeberry','hedgehog','helicopter','kangaroo','key','lantern','lighthouse','lion','lionsheep','lobster','map','mermaid','monapassport','monkey','mosquito','octopus','owl','paintbrush','palm_tree','parrot','passport','peas','penguin','pig','pigsheep','pineapple','pool','postcard','power_outlet','rabbit','rabbitturtle','radio','radioface','rain','rhinoceros','rifle','roller_coaster','sandwich','scorpion','sea_turtle','sheep','skull','snail','snowflake','speedboat','spider','squirrel','steak','stove','strawberry','swan','swing_set','the_mona_lisa','tiger','toothbrush','toothpaste','tractor','trombone','truck','whale','windmill','yoga','yogabicycle']
BASE_URL = "https://storage.googleapis.com/quickdraw-models/sketchRNN/models/"

=======
>>>>>>> origin/9-integrating-with-streamlit-ui
## PLACEHOLDER MODEL CLASS (to be replaced by ML/model.py's SketchRNN - same method names)
class ShadowModel:
    def __init__(self, url): self.url = url
    def initialize(self): pass
    def setPixelFactor(self, f): pass
    def zeroState(self): return {"c": [0], "h": [0]}
    def copyState(self, s): return dict(s)
    def update(self, stroke, state): return state
    def updateStrokes(self, strokes, state, steps): return state
    def getPDF(self, state, temperature): return {}
    def zeroInput(self): return [0, 0, 1, 0, 0]

    def sample(self, pdf):
        """Fake: a random small move"""
        if random.random() < 0.05:
            return [0, 0, 0, 0, 1]
        return [random.uniform(-10, 10), random.uniform(-10, 10), 1, 0, 0]

    def simplify_line(self, line, tolerance=2.0):
        return line

    def line_to_stroke(self, line, last_point):
        """ Absolute points -> [dx, dy, p_down, p_up, p_end] offsets.
        Pen is down for every point except the last, where it lifts."""
        strokes = []
        px, py = last_point
        for j, (x, y) in enumerate(line):
            last = j == len(line) - 1
            strokes.append([x - px, y - py, 0 if last else 1, 1 if last else 0, 0])
            px, py = x, y
        return strokes

SketchRNN = ShadowModel   # later: from ML.model import SketchRNN

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
