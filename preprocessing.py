"""
preprocessing.py
-----------------
SINGLE SOURCE OF TRUTH for how an X-ray image is turned into a model-ready
array. Both the training pipeline (notebook / train.py) and the Streamlit
app (app.py) import THIS file and call THESE functions.

Why this file exists (and why it matters):
    The #1 reason a "it worked in training but the app gives wrong answers"
    bug happens is TRAIN/INFERENCE SKEW: the resize method, pixel scaling,
    color channel order, or class-index mapping used at inference time is
    slightly different from what the model was trained on. Because both
    train.py/the notebook and app.py import this module instead of each
    re-implementing their own resize/normalize code, that class of bug is
    structurally impossible here.

Everything a downstream script needs to know about the data (image size,
class names, class order) is defined ONCE, right here, as constants.
"""

import os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS - change these in exactly one place and every script
# (notebook, train.py, app.py) picks up the change automatically.
# ---------------------------------------------------------------------------

# Target size every image is resized to before it touches a model.
# 128x128 keeps training fast on CPU while still being big enough for a
# CNN (and for transfer-learning backbones, which accept any size >= 32x32
# when include_top=False) to pick up meaningful lung texture patterns.
IMG_SIZE = (128, 128)  # (height, width)

# Fixed, alphabetically-sorted class order. This MUST match the order
# Keras' image_dataset_from_directory / ImageDataGenerator assigns
# (both sort class subfolder names alphabetically), so the index a model
# outputs (argmax) always maps to the same human-readable label.
CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]

# Folder names inside the dataset root (must match the unzipped Kaggle
# dataset exactly, including the space in "Viral Pneumonia").
TRAIN_DIR_NAME = "train"
TEST_DIR_NAME = "test"

# Valid image extensions we will accept when scanning folders.
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_dataset_root(candidates):
    """
    Look through a list of candidate paths and return the first one that
    actually contains a 'train' and 'test' subfolder.

    Why: different environments (Colab after unzipping a Kaggle download,
    a local clone, this project's own 'data/' folder) end up with the
    dataset at slightly different paths. Rather than hard-coding one path
    everywhere, every script calls this once and gets back a working path
    or a clear error - instead of a confusing FileNotFoundError three
    layers deep inside Keras.
    """
    for path in candidates:
        train_p = os.path.join(path, TRAIN_DIR_NAME)
        test_p = os.path.join(path, TEST_DIR_NAME)
        if os.path.isdir(train_p) and os.path.isdir(test_p):
            return path
    raise FileNotFoundError(
        "Could not find the Covid19-dataset folder (expected a 'train' and "
        "'test' subfolder). Checked: " + ", ".join(candidates) +
        ". Update DATASET_CANDIDATES at the top of the calling script."
    )


def load_and_preprocess_image(image_path_or_bytes):
    """
    Load ONE image (from a file path, an opened file, or raw bytes) and
    turn it into the exact array shape/scale the model expects.

    Returns
    -------
    np.ndarray of shape (1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype float32,
    values scaled to [0, 1]. The leading batch dimension of 1 means the
    output can be passed straight to model.predict(...).

    This is the ONLY function allowed to do resizing/scaling. If you ever
    need to change how images are preprocessed, change it here - training
    and the Streamlit app will both pick up the change identically.
    """
    img = Image.open(image_path_or_bytes)
    img = img.convert("RGB")  # collapse grayscale/PNG-with-alpha/etc. to 3 channels
    img = img.resize(IMG_SIZE)  # (width, height) order, matches PIL convention
    arr = np.asarray(img, dtype=np.float32) / 255.0  # scale 0-255 -> 0-1
    arr = np.expand_dims(arr, axis=0)  # add batch dimension -> (1, H, W, 3)
    return arr


def load_dataset_from_folders(root_dir, class_names=CLASS_NAMES, img_size=IMG_SIZE):
    """
    Walk root_dir/<class_name>/*.jpg|png and load every image into memory
    as numpy arrays. Used for the 'train' and 'test' splits.

    Returns
    -------
    X : np.ndarray, shape (N, H, W, 3), float32, scaled to [0, 1]
    y : np.ndarray, shape (N,), integer label = index into class_names
    paths : list[str] of length N, the original file path of each image
            (kept around so we can display filenames in EDA plots).

    Why load fully into memory instead of streaming with a generator?
    The whole dataset is ~300 images total, so this comfortably fits in
    RAM and keeps every downstream step (EDA, class-imbalance analysis,
    train/val/test splitting, feeding into ImageDataGenerator.flow(),
    Keras Tuner) simple: it's all just numpy arrays.
    """
    X, y, paths = [], [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(VALID_EXTENSIONS):
                continue
            fpath = os.path.join(class_dir, fname)
            img = Image.open(fpath).convert("RGB").resize(img_size)
            X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(label_idx)
            paths.append(fpath)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), paths
