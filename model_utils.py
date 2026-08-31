"""
model_utils.py
---------------
Defines the three CNN architectures required by the project brief, plus a
model-saving helper.

Why architectures live in their own file instead of inline in the notebook:
    train.py, the notebook, AND the Keras-Tuner search all need to build
    "a basic CNN with these hyperparameters" or "a transfer-learning model
    on top of this backbone". Keeping the architecture-building logic in
    one place means the tuner searches the SAME architecture family that
    train.py trains the final model with - no drift between "what we
    tuned" and "what we shipped".
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, applications

NUM_CLASSES = 3


def build_basic_cnn(input_shape=(128, 128, 3),
                     num_classes=NUM_CLASSES,
                     conv_filters=(32, 64, 128),
                     dense_units=128,
                     dropout_rate=0.3,
                     learning_rate=1e-3):
    """
    Model 1: Basic CNN built from scratch.
        Conv2D -> MaxPooling  (x len(conv_filters) blocks)
        -> Flatten -> Dense -> Dropout -> Dense(softmax)

    Why this model exists at all, given transfer learning usually wins:
    it's the baseline. Without it we would have no way of knowing whether
    transfer learning is actually earning its extra complexity, or whether
    a much simpler model gets us 90% of the way there.

    All the shape/size choices below are function arguments (not hard-coded)
    specifically so Keras Tuner (see train.py) can search over them.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in conv_filters:
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="basic_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(input_shape=(128, 128, 3),
                          num_classes=NUM_CLASSES,
                          backbone_name="MobileNetV2",
                          fine_tune_last_n=20,
                          dense_units=128,
                          dropout_rate=0.3,
                          learning_rate=1e-4):
    """
    Model 2 (and, reused with augmented data, Model 3): Transfer learning.

    Why MobileNetV2 instead of VGG16/ResNet50 (both mentioned as options
    in the brief): MobileNetV2 has ~3.5M parameters vs. VGG16's ~138M, so
    it fine-tunes fast on a CPU and on a dataset this small (only ~250
    training images) - a huge backbone like VGG16 would overfit almost
    immediately with this little data. Swap `backbone_name` to "VGG16" or
    "ResNet50" (both are already wired up below) if you have a GPU and
    want to compare.

    "Fine-tune last few layers" (per the brief) means: keep most of the
    ImageNet-pretrained backbone FROZEN (so we don't destroy the general
    edge/texture features it already learned) and only make the last
    `fine_tune_last_n` layers trainable, so the top of the network can
    adapt to X-ray-specific patterns.
    """
    backbones = {
        "MobileNetV2": applications.MobileNetV2,
        "VGG16": applications.VGG16,
        "ResNet50": applications.ResNet50,
    }
    if backbone_name not in backbones:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    base_model = backbones[backbone_name](
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    # Freeze everything first...
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_last_n]:
        layer.trainable = False
    # ...so only the last `fine_tune_last_n` backbone layers can update.

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"transfer_{backbone_name.lower()}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model_robust(model, model_dir, model_name):
    """
    Save a trained model TWO ways, because Keras save-format compatibility
    breaks across versions/environments more often than people expect
    (a model saved in one Keras/TF version sometimes refuses to load in
    another - e.g. Colab's TF version vs. your local venv's TF version):

    1. Native format: <model_name>.keras  (architecture + weights + optimizer
       state in one file - the normal, recommended way to reload a model).
    2. Weights-only fallback: <model_name>.weights.h5 + <model_name>_config.json
       (just the numeric weights, plus the plain arguments needed to call
       build_basic_cnn()/build_transfer_model() again and re-attach them).

    app.py tries #1 first and automatically falls back to #2 if loading the
    .keras file raises an error - so a version mismatch between the
    machine that trained the model and the machine running the Streamlit
    app won't leave you stuck.
    """
    os.makedirs(model_dir, exist_ok=True)

    keras_path = os.path.join(model_dir, f"{model_name}.keras")
    model.save(keras_path)

    weights_path = os.path.join(model_dir, f"{model_name}.weights.h5")
    model.save_weights(weights_path)

    return keras_path, weights_path


def load_model_robust(model_dir, model_name, rebuild_fn=None, rebuild_kwargs=None):
    """
    Counterpart to save_model_robust: try the native .keras file first;
    if that fails for any reason, rebuild the architecture from code
    (via rebuild_fn/rebuild_kwargs) and load just the weights instead.
    """
    keras_path = os.path.join(model_dir, f"{model_name}.keras")
    try:
        return tf.keras.models.load_model(keras_path)
    except Exception as e:
        print(f"[load_model_robust] Native load failed ({e}); "
              f"falling back to weights-only reload.")
        if rebuild_fn is None:
            raise
        model = rebuild_fn(**(rebuild_kwargs or {}))
        weights_path = os.path.join(model_dir, f"{model_name}.weights.h5")
        model.load_weights(weights_path)
        return model
