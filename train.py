"""
train.py
========
End-to-end training pipeline for COVID-19 chest X-ray classification.

Run this from the project folder with:
    python train.py

What it does, in order (mirrors the notebook's task numbering exactly, so
you can cross-reference explanations there):
    1. Data loading & exploration
    2. Preprocessing (normalize, encode labels, train/val/test split)
    3. EDA (class distribution, sample images) -> saved into outputs/
    4. Train 3 CNN models (basic CNN, transfer learning, transfer + augmentation)
    5. Evaluate all 3 (accuracy, confusion matrix, classification report, ROC-AUC)
    6. Class imbalance analysis + class-weighted training
    7. Hyperparameter tuning of the basic CNN with Keras Tuner
    8. Model comparison table -> saved into outputs/model_comparison.csv
    9. Save the best-performing model into models/ for the Streamlit app

Everything that prints a plot ALSO saves it as a PNG into outputs/, so you
have a permanent record even if you're running this headless.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; we only save PNGs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import (
    IMG_SIZE, CLASS_NAMES, find_dataset_root, load_dataset_from_folders,
)
from model_utils import build_basic_cnn, build_transfer_model, save_model_robust

# ---------------------------------------------------------------------------
# CONFIG - tune these based on your hardware. The values below are kept
# small on purpose so the whole pipeline finishes in a few minutes on a
# laptop CPU; raise EPOCHS / TUNER_MAX_TRIALS if you have a GPU.
# ---------------------------------------------------------------------------
DATASET_CANDIDATES = [
    "data/Covid19-dataset",
    "Covid19-dataset",
    "./Covid19-dataset",
]
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
RANDOM_STATE = 42
VAL_SPLIT = 0.2          # fraction of TRAIN data held out for validation
EPOCHS = 15
BATCH_SIZE = 16
TUNER_MAX_TRIALS = 5
TUNER_EPOCHS = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# TASK 1-2: Data loading, exploration & preprocessing
# ---------------------------------------------------------------------------
section("TASK 1-2: Data loading, exploration & preprocessing")

dataset_root = find_dataset_root(DATASET_CANDIDATES)
print(f"Dataset found at: {dataset_root}")

X_train_full, y_train_full, train_paths = load_dataset_from_folders(
    os.path.join(dataset_root, "train"), CLASS_NAMES, IMG_SIZE
)
X_test, y_test, test_paths = load_dataset_from_folders(
    os.path.join(dataset_root, "test"), CLASS_NAMES, IMG_SIZE
)

print(f"Train+val images: {X_train_full.shape[0]}  |  Test images: {X_test.shape[0]}")
print(f"Image tensor shape: {X_train_full.shape[1:]} (already normalized to [0,1])")

# Why stratify: with only ~250 training images split three ways, a random
# split could easily starve the validation set of one class entirely.
# Stratifying keeps the same class proportions in train and val.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VAL_SPLIT, stratify=y_train_full, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Dataset size per class (train+val combined, then test)
counts_train = pd.Series(y_train_full).value_counts().sort_index()
counts_test = pd.Series(y_test).value_counts().sort_index()
print("\nClass counts (train+val split):")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  {name:<18} train+val={counts_train.get(idx,0):<4} test={counts_test.get(idx,0)}")


# ---------------------------------------------------------------------------
# TASK 3: EDA
# ---------------------------------------------------------------------------
section("TASK 3: Exploratory Data Analysis")

# Why we look at this before modeling: class imbalance and visually similar
# classes (viral pneumonia and COVID often look alike on an X-ray) are the
# two things most likely to explain a confusing confusion matrix later, so
# it's worth seeing them up front.
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=[CLASS_NAMES[i] for i in counts_train.index], y=counts_train.values, ax=ax)
ax.set_title("Training set class distribution")
ax.set_ylabel("Number of images")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=120)
plt.close()

fig, axes = plt.subplots(len(CLASS_NAMES), 4, figsize=(12, 3 * len(CLASS_NAMES)))
rng = np.random.default_rng(RANDOM_STATE)
for row, class_idx in enumerate(range(len(CLASS_NAMES))):
    idxs = np.where(y_train_full == class_idx)[0]
    sample_idxs = rng.choice(idxs, size=min(4, len(idxs)), replace=False)
    for col, s_idx in enumerate(sample_idxs):
        axes[row, col].imshow(X_train_full[s_idx])
        axes[row, col].set_title(CLASS_NAMES[class_idx], fontsize=9)
        axes[row, col].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"), dpi=120)
plt.close()
print(f"Saved class_distribution.png and sample_images.png to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# TASK 6 (computed early, used by training below): Class imbalance
# ---------------------------------------------------------------------------
section("TASK 6: Class imbalance analysis")

# Why class_weight instead of naive oversampling: with only 250 images,
# duplicating minority-class images verbatim (oversampling) risks the
# model memorizing exact pixels rather than learning general features.
# class_weight achieves the same "pay more attention to rare classes"
# effect purely through the loss function, with zero duplicate images -
# and we ALSO use augmentation (Task 4, Model 3) as a complementary,
# non-duplicating way to grow the effective size of minority classes.
class_weights_array = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
print("Class weights (balanced):", {CLASS_NAMES[k]: round(v, 3) for k, v in class_weight_dict.items()})


# ---------------------------------------------------------------------------
# TASK 4: CNN model building - Model 1 (Basic CNN)
# ---------------------------------------------------------------------------
section("TASK 4a: Model 1 - Basic CNN")

model1 = build_basic_cnn(input_shape=X_train.shape[1:])
model1.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

history1 = model1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=2,
)


# ---------------------------------------------------------------------------
# TASK 4: CNN model building - Model 2 (Transfer learning, no augmentation)
# ---------------------------------------------------------------------------
section("TASK 4b: Model 2 - Transfer learning (MobileNetV2)")

model2 = build_transfer_model(input_shape=X_train.shape[1:], backbone_name="MobileNetV2")
model2.summary()

history2 = model2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=2,
)


# ---------------------------------------------------------------------------
# TASK 4 + 6: CNN model building - Model 3 (Transfer learning + augmentation)
# ---------------------------------------------------------------------------
section("TASK 4c: Model 3 - Transfer learning + data augmentation")

# Why augmentation here specifically: this is the "oversampling via data
# augmentation" technique named in the brief's class-imbalance task. Rather
# than literally duplicating minority-class files, ImageDataGenerator
# creates realistic variations (small rotations/shifts/zooms/flips) of
# EVERY image on the fly each epoch, which both fights overfitting and -
# combined with class_weight - directly addresses the imbalance.
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)
train_datagen.fit(X_train)
train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_STATE)
# NOTE: train_gen already knows its own length (it's a keras.utils.Sequence
# under the hood), so we deliberately do NOT pass steps_per_epoch here.
# Explicitly passing steps_per_epoch alongside a Sequence-based generator
# under Keras 3 causes training to silently stop consuming batches after
# the first epoch ("input ran out of data") - omitting it lets Keras use
# len(train_gen) automatically and training proceeds normally every epoch.

model3 = build_transfer_model(input_shape=X_train.shape[1:], backbone_name="MobileNetV2")

history3 = model3.fit(
    train_gen,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=2,
)


# ---------------------------------------------------------------------------
# TASK 5: Model evaluation (shared function so all 3 models are scored
# identically and fairly)
# ---------------------------------------------------------------------------
section("TASK 5: Model evaluation on the held-out TEST set")


def evaluate_model(model, X, y_true, name):
    """
    Computes every metric the brief asks for, for one model, on one
    dataset split. Returns a dict so results can be assembled into the
    Task 8 comparison table.
    """
    y_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")  # can happen if a class is missing from y_true

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision(macro): {precision:.4f} | "
          f"Recall(macro): {recall:.4f} | F1(macro): {f1:.4f} | ROC-AUC(macro): {roc_auc:.4f}")
    print(report)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_name}.png"), dpi=120)
    plt.close()

    return {
        "model": name,
        "test_accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "roc_auc_macro": roc_auc,
        "trainable_params": int(np.sum([np.prod(v.shape) for v in model.trainable_weights])),
        "total_params": model.count_params(),
    }


def plot_training_curves(history, name):
    """
    Task 5 also asks us to check for overfitting via train-vs-val curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"training_curves_{safe_name}.png"), dpi=120)
    plt.close()


results = []
results.append(evaluate_model(model1, X_test, y_test, "Model 1 - Basic CNN"))
results.append(evaluate_model(model2, X_test, y_test, "Model 2 - Transfer Learning"))
results.append(evaluate_model(model3, X_test, y_test, "Model 3 - Transfer + Augmentation"))

plot_training_curves(history1, "Model 1 - Basic CNN")
plot_training_curves(history2, "Model 2 - Transfer Learning")
plot_training_curves(history3, "Model 3 - Transfer + Augmentation")


# ---------------------------------------------------------------------------
# TASK 7: Hyperparameter tuning (Keras Tuner) on the Basic CNN
# ---------------------------------------------------------------------------
section("TASK 7: Hyperparameter tuning with Keras Tuner")

# Why we tune the Basic CNN specifically (not the transfer-learning models):
# tuning is about searching architecture-level choices (filter counts,
# dropout, learning rate). The transfer-learning models' architecture is
# mostly fixed by the pretrained backbone, so there's much less to search -
# the basic CNN is where hyperparameter choices actually move the needle.
try:
    import keras_tuner as kt

    def build_hp_model(hp):
        conv_filters = (
            hp.Choice("filters_1", [16, 32, 64]),
            hp.Choice("filters_2", [32, 64, 128]),
            hp.Choice("filters_3", [64, 128, 256]),
        )
        dense_units = hp.Choice("dense_units", [64, 128, 256])
        dropout_rate = hp.Float("dropout_rate", 0.2, 0.5, step=0.1)
        learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        return build_basic_cnn(
            input_shape=X_train.shape[1:],
            conv_filters=conv_filters,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )

    tuner = kt.RandomSearch(
        build_hp_model,
        objective="val_accuracy",
        max_trials=TUNER_MAX_TRIALS,
        executions_per_trial=1,
        overwrite=True,
        directory=os.path.join(OUTPUT_DIR, "kt_search"),
        project_name="covid_cnn_tuning",
    )
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=TUNER_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=2,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparameters found:", best_hp.values)

    tuned_model = tuner.hypermodel.build(best_hp)
    history_tuned = tuned_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=2,
    )
    tuned_result = evaluate_model(tuned_model, X_test, y_test, "Model 4 - Tuned Basic CNN")
    plot_training_curves(history_tuned, "Model 4 - Tuned Basic CNN")
    results.append(tuned_result)

    all_models = {
        "Model 1 - Basic CNN": model1,
        "Model 2 - Transfer Learning": model2,
        "Model 3 - Transfer + Augmentation": model3,
        "Model 4 - Tuned Basic CNN": tuned_model,
    }
except ImportError:
    print("keras_tuner not installed - skipping Task 7. "
          "Install with: pip install keras-tuner")
    all_models = {
        "Model 1 - Basic CNN": model1,
        "Model 2 - Transfer Learning": model2,
        "Model 3 - Transfer + Augmentation": model3,
    }


# ---------------------------------------------------------------------------
# TASK 8: Model comparison table
# ---------------------------------------------------------------------------
section("TASK 8: Model comparison table")

comparison_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print(comparison_df.to_string(index=False))


# ---------------------------------------------------------------------------
# TASK 9 (prep): Save the best model for the Streamlit app
# ---------------------------------------------------------------------------
section("Saving best model for the Streamlit app")

best_row = comparison_df.iloc[0]
best_model_name = best_row["model"]
best_model = all_models[best_model_name]
print(f"Best model by test accuracy: {best_model_name} "
      f"(accuracy={best_row['test_accuracy']:.4f})")

keras_path, weights_path = save_model_robust(best_model, MODEL_DIR, "best_model")

# --- NEW: Export to TFLite (both float32 and int8 quantized) ---
import tensorflow as tf

# Float32 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()
with open(os.path.join(MODEL_DIR, "best_model.tflite"), "wb") as f:
    f.write(tflite_model)
print(f"Saved float32 TFLite model to: {MODEL_DIR}/best_model.tflite")

# Int8 quantized TFLite
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quant_tflite_model = converter.convert()
with open(os.path.join(MODEL_DIR, "best_model_int8.tflite"), "wb") as f:
    f.write(quant_tflite_model)
print(f"Saved int8 quantized TFLite model to: {MODEL_DIR}/best_model_int8.tflite")

# Save metadata
metadata = {
    "best_model_name": best_model_name,
    "img_size": list(IMG_SIZE),
    "class_names": CLASS_NAMES,
    "test_accuracy": float(best_row["test_accuracy"]),
    "is_transfer_model": "Transfer" in best_model_name,
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved model to: {keras_path}")
print(f"Saved weights fallback to: {weights_path}")
print(f"Saved metadata to: {os.path.join(MODEL_DIR, 'metadata.json')}")
print("\nDONE. You can now run:  streamlit run app.py"

print(f"\nSaved model to: {keras_path}")
print(f"Saved weights fallback to: {weights_path}")
print(f"Saved metadata to: {os.path.join(MODEL_DIR, 'metadata.json')}")
print("\nDONE. You can now run:  streamlit run app.py")
