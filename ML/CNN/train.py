"""Train a CNN classifier on QuickDraw bitmaps rasterized by prepare.py.

Usage:
    python prepare.py          # one-time: cache rasterized bitmaps
    python train.py            # trains and saves data/model.keras
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from rasterize import IMG_SIZE


ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
CLASSES_FILE = ROOT / "data" / "classes.txt"
MODEL_PATH = ROOT / "data" / "model.keras"

VAL_FRACTION = 0.1


def load_cached(classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for idx, cat in enumerate(classes):
        path = CACHE_DIR / f"{cat}_{IMG_SIZE}.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing cache for {cat!r}; run prepare.py first")
        arr = np.load(path)
        print(f"  {cat}: {arr.shape[0]} samples")
        xs.append(arr)
        ys.append(np.full(arr.shape[0], idx, dtype=np.int32))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y


def shuffle_split(x: np.ndarray, y: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    n_val = int(len(x) * VAL_FRACTION)
    return (x[n_val:], y[n_val:]), (x[:n_val], y[:n_val])


def build_model(num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Rescaling(1.0 / 255)(inp)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_dataset(x: np.ndarray, y: np.ndarray, batch: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x[..., None], y))
    if training:
        ds = ds.shuffle(10_000, seed=0)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap batches per epoch (quick smoke tests). omit for a full pass.")
    args = ap.parse_args()

    classes = CLASSES_FILE.read_text().strip().splitlines()
    print(f"classes: {classes}")

    print("loading cached bitmaps...")
    x, y = load_cached(classes)
    print(f"total: {x.shape[0]} samples of {x.shape[1:]}")

    (x_tr, y_tr), (x_val, y_val) = shuffle_split(x, y)
    print(f"train: {len(x_tr)}  val: {len(x_val)}")

    train_ds = make_dataset(x_tr, y_tr, args.batch, training=True)
    val_ds = make_dataset(x_val, y_val, args.batch, training=False)

    model = build_model(len(classes))
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_accuracy"),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=args.max_steps,
        callbacks=cbs,
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nfinal val_acc: {val_acc:.4f}  val_loss: {val_loss:.4f}")
    print(f"saved model -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
