"""Generate a sketch with the SketchRNN model, show it, and score it.

Pipeline:
    SketchRNN.sample loop  ->  stroke-5 sequence
      -> convert to stroke-3
      -> rasterize with the SAME rasterizer used to train the classifier
      -> run data/model.keras and read p(target)

Usage:
    python generate_and_evaluate.py                          # target=cat
    python generate_and_evaluate.py --target cat --temperature 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import model as sketchrnn_mod  # noqa: E402  (module-level instance is `sketchy`)
from rasterize import IMG_SIZE, rasterize, strokes_to_polylines  # noqa: E402

SketchRNN = sketchrnn_mod.SketchRNN

PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2
MODEL_PATH = ROOT / "data" / "model.keras"
CLASSES_FILE = ROOT / "data" / "classes.txt"


def generate_sketch(
    model: SketchRNN,
    temperature: float = 0.4,
    max_steps: int = 500,
) -> np.ndarray:
    """Sample a sketch from SketchRNN, returned as stroke-5 array."""
    state = model.zeroState()
    # Seed: a single pen-down "start" stroke, same as interactive_predict.py.
    state = model.update([0, 0, 1, 0, 0], state)

    strokes: list[list[float]] = []
    dx, dy = 0.0, 0.0
    prev_pen = [1, 0, 0]  # penDown

    for _ in range(max_steps):
        state = model.update([dx, dy] + prev_pen, state)
        pdf = model.getPDF(state, temperature)
        sample = model.sample(pdf)

        dx, dy = float(sample[0]), float(sample[1])
        pen = [int(sample[2]), int(sample[3]), int(sample[4])]

        strokes.append([dx, dy, pen[0], pen[1], pen[2]])
        prev_pen = pen

        if pen[PEN_END] == 1:
            break

    return np.asarray(strokes, dtype=np.float32)


def stroke5_to_stroke3(strokes5: np.ndarray) -> np.ndarray:
    """Convert stroke-5 [dx, dy, pDown, pUp, pEnd] to stroke-3 [dx, dy, pen_lift]."""
    out = np.zeros((len(strokes5), 3), dtype=np.int32)
    out[:, 0] = np.round(strokes5[:, 0])
    out[:, 1] = np.round(strokes5[:, 1])
    # pen_lift = 1 whenever this is the last point of a stroke (pen up OR end).
    out[:, 2] = ((strokes5[:, 3] == 1) | (strokes5[:, 4] == 1)).astype(np.int32)
    return out


def classify(bitmap: np.ndarray, classes: list[str]):
    """Return (probs vector, model) or (None, None) if no trained model yet."""
    if not MODEL_PATH.exists():
        print(f"(no trained classifier at {MODEL_PATH} — skipping scoring)")
        return None
    import tensorflow as tf  # imported lazily so generation works without TF-trained model
    clf = tf.keras.models.load_model(MODEL_PATH)
    x = bitmap.astype(np.float32)[None, ..., None]
    probs = clf.predict(x, verbose=0)[0]
    return probs


def plot_result(strokes3: np.ndarray, bitmap: np.ndarray, probs, classes, target: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # --- left: vector sketch ---
    ax = axes[0]
    polylines = strokes_to_polylines(strokes3)
    for line in polylines:
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        ax.plot(xs, ys, color="black", linewidth=2)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # SketchRNN y-axis grows downward
    ax.set_title(f"generated ({len(polylines)} strokes)")
    ax.axis("off")

    # --- middle: rasterized bitmap seen by the classifier ---
    axes[1].imshow(bitmap, cmap="gray_r", interpolation="nearest")
    axes[1].set_title(f"{IMG_SIZE}x{IMG_SIZE} bitmap (classifier input)")
    axes[1].axis("off")

    # --- right: probability bar chart ---
    ax = axes[2]
    if probs is None:
        ax.text(0.5, 0.5, "train the classifier\nfirst:\npython train.py",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
    else:
        order = np.argsort(probs)[::-1]
        names = [classes[i] for i in order]
        colors = ["#2a9d8f" if n == target else "#bbbbbb" for n in names]
        ax.barh(range(len(names)), [probs[i] for i in order], color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        if target in classes:
            p_target = float(probs[classes.index(target)])
            ax.set_title(f"p({target}) = {p_target:.1%}")
        else:
            ax.set_title("(target not in class list)")

    plt.tight_layout()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cat", help="which class to score against")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", type=Path, default=None, help="save figure to this path")
    args = ap.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)

    print("loading SketchRNN...")
    # model/model.py already instantiates + initializes `sketchy` at import time.
    model = sketchrnn_mod.sketchy
    model.setPixelFactor(2.0)

    print(f"sampling (temp={args.temperature}, max_steps={args.max_steps})...")
    strokes5 = generate_sketch(model, args.temperature, args.max_steps)
    print(f"got {len(strokes5)} stroke-5 steps")

    strokes3 = stroke5_to_stroke3(strokes5)
    bitmap = rasterize(strokes3)

    classes: list[str] = []
    if CLASSES_FILE.exists():
        classes = CLASSES_FILE.read_text().strip().splitlines()

    probs = classify(bitmap, classes) if classes else None

    if probs is not None and args.target in classes:
        p = float(probs[classes.index(args.target)])
        print(f"\np({args.target}) = {p:.4f}   ({p:.1%})")
        top3 = np.argsort(probs)[-3:][::-1]
        print("top-3: " + ", ".join(f"{classes[i]}={probs[i]:.2f}" for i in top3))

    fig = plot_result(strokes3, bitmap, probs, classes, args.target)
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved -> {args.save}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
