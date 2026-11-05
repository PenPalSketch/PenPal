import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .model import SketchRNN


NUM_TURNS = 20
TEMPERATURE = 0.25


def generate_stroke(model, state, position, pen_down):
    """Generate one stroke and return it with the updated state and position."""
    # Group the model's movements into one stroke so we can count turns.
    # Streamlit draws each movement as it goes, using st.rerun().
    x, y = position
    points, samples = [], []
    for _ in range(500):
        pdf = model.getPDF(state, TEMPERATURE)
        # Disable end-of-drawing so the demo reaches NUM_TURNS strokes.
        pdf["pen"][2] = 0
        pdf["pen"] /= pdf["pen"].sum()
        sample = model.sample(pdf)
        dx, dy = float(sample[0]), float(sample[1])
        next_x, next_y = x + dx, y + dy

        # The previous pen state controls whether this movement draws.
        if pen_down:
            if not points:
                points.append([x, y])
            points.append([next_x, next_y])

        x, y = next_x, next_y
        pen_down = bool(sample[2])
        samples.append([dx, dy, *sample[2:]])
        state = model.update(sample, state)
        if points and not pen_down:
            stroke = {"points": points, "samples": samples, "finished": False}
            return stroke, state, (x, y)

    raise RuntimeError("Stroke did not finish within 500 steps.")


def generate_turns(model):
    state = model.update(model.zero_input(), model.zeroState())
    position = (0.0, 0.0)
    pen_down = True
    history = []

    for turn in range(1, NUM_TURNS + 1):
        stroke, state, position = generate_stroke(model, state, position, pen_down)
        pen_down = False  # Each completed stroke ends with the pen lifted.
        history.append({"turn": turn, **stroke})
        print(f"Turn {turn}/{NUM_TURNS}: {len(stroke['points']) - 1} drawn segments", flush=True)

    return history


def main():
    random.seed(7)
    model = SketchRNN("cat")
    model.initialize()
    model.setPixelFactor(5.0)
    try:
        turns = generate_turns(model)
    finally:
        model.dispose()

    save_demo(turns)


# === AI administrative functions ===
# the following functions below are not super important to understand for a human.
# just let the agent read them.
# they handle administrative stuff like centering the strokes on the final output,
# saving it as a GIF etc,.


def fit_strokes_to_canvas(turns):
    """Fit all strokes together so the drawing stays still during playback."""
    points = [point for turn in turns for point in turn["points"]]
    xs, ys = zip(*points)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    scale = 520 / span
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2

    strokes = []
    for turn in turns:
        strokes.append([((x - center_x) * scale + 320, (y - center_y) * scale + 350)
                        for x, y in turn["points"]])
    return strokes


def make_frame(canvas, label):
    snapshot = canvas.copy()
    text = ImageDraw.Draw(snapshot)
    text.text((20, 14), "PenPal: model self-play", fill="black", font_size=24)
    text.text((20, 46), label, fill="black", font_size=18)
    text.text((20, 615), "No seed stroke. End-of-drawing disabled for this demo.",
              fill="#555555", font_size=14)
    return snapshot


def save_animation(turns, destination):
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706",
              "#0891b2", "#db2777", "#4f46e5", "#65a30d", "#475569"]
    canvas = Image.new("RGB", (640, 640), "white")
    drawing = ImageDraw.Draw(canvas)
    frames = [make_frame(canvas, f"Ready: 0 / {len(turns)} turns")]
    durations = [800]

    for turn, points in enumerate(fit_strokes_to_canvas(turns), 1):
        color = colors[(turn - 1) % len(colors)]
        for start, end in zip(points, points[1:]):
            drawing.line([start, end], fill=color, width=3)
            frames.append(make_frame(canvas, f"Turn {turn} / {len(turns)}"))
            durations.append(60)
        durations[-1] = 700
    frames.append(make_frame(canvas, f"Complete: {len(turns)} / {len(turns)} turns"))
    durations.append(2500)
    frames[0].save(destination, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)


def save_demo(turns):
    output = Path(__file__).parent / "output" / "self_play.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_animation(turns, output)
    data = {"model": "cat", "random_seed": 7,
            "temperature": TEMPERATURE, "end_of_drawing_disabled": True,
            "turns": turns}
    output.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Animation: {output.resolve()}")
    print(f"Stroke data: {output.with_suffix('.json').resolve()}")

# === end AI administrative functions ===

if __name__ == "__main__":
    main()
