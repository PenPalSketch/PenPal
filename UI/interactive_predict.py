import sys
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pen index constants
PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2

AVAILABLE_MODELS = ['bird', 'ant','ambulance','angel','alarm_clock','antyoga','backpack','barn','basket','bear','bee','beeflower','bicycle','book','brain','bridge','bulldozer','bus','butterfly','cactus','calendar','castle','cat','catbus','catpig','chair','couch','crab','crabchair','crabrabbitfacepig','cruise_ship','diving_board','dog','dogbunny','dolphin','duck','elephant','elephantpig','everything','eye','face','fan','fire_hydrant','firetruck','flamingo','flower','floweryoga','frog','frogsofa','garden','hand','hedgeberry','hedgehog','helicopter','kangaroo','key','lantern','lighthouse','lion','lionsheep','lobster','map','mermaid','monapassport','monkey','mosquito','octopus','owl','paintbrush','palm_tree','parrot','passport','peas','penguin','pig','pigsheep','pineapple','pool','postcard','power_outlet','rabbit','rabbitturtle','radio','radioface','rain','rhinoceros','rifle','roller_coaster','sandwich','scorpion','sea_turtle','sheep','skull','snail','snowflake','speedboat','spider','squirrel','steak','stove','strawberry','swan','swing_set','the_mona_lisa','tiger','toothbrush','toothpaste','tractor','trombone','truck','whale','windmill','yoga','yogabicycle']
BASE_URL = "https://storage.googleapis.com/quickdraw-models/sketchRNN/models/"

from ML.model import SketchRNN

# same as js's initModel()
def load_model(name):
    model = SketchRNN(f"{BASE_URL}{name}.gen.json")
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