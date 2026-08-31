# COVID-19 Chest X-ray Classification — End-to-End Project

Classifies a chest X-ray into **Covid**, **Normal**, or **Viral Pneumonia**
using CNNs (a from-scratch model + two MobileNetV2 transfer-learning
variants), with a Streamlit app for live predictions.

Every file in this project was **actually run end-to-end** while building
it (full training pipeline + notebook execution + Streamlit app boot +
a real prediction) — not just written and assumed to work. Two real bugs
were caught and fixed this way (see "Notes on gotchas already fixed" below).

## Project structure

```
covid19_project/
├── data/
│   └── Covid19-dataset/
│       ├── train/{Covid,Normal,Viral Pneumonia}/*.jpg|png
│       └── test/{Covid,Normal,Viral Pneumonia}/*.jpg|png
├── preprocessing.py        # single source of truth for image resize/scale + data loading
├── model_utils.py          # 3 CNN architectures + robust save/load helper
├── train.py                # full pipeline script: run this first
├── COVID19_Xray_Classification.ipynb   # same pipeline, notebook form, with "why" explanations
├── app.py                  # Streamlit app (run after train.py / the notebook)
├── requirements.txt
├── models/                 # created by train.py / the notebook — best_model.keras etc.
└── outputs/                # created by train.py / the notebook — plots + comparison table
```

## Setup (Windows CMD, per your usual workflow)

```cmd
cd path\to\covid19_project
python -m venv covid19_venv
covid19_venv\Scripts\activate
pip install -r requirements.txt
```

## Run order — this matters

**1. Train the models first** (creates `models/best_model.keras`, which the
app needs to exist before it can run):

```cmd
python train.py
```

This takes a few minutes on CPU (the dataset is small — ~250 training
images). It will print progress for each of the 9 project tasks and save
all plots into `outputs/`.

*(Equivalently, you can open `COVID19_Xray_Classification.ipynb` in
Jupyter/VS Code and run all cells top to bottom — it does the same thing
with explanatory markdown before every step.)*

**2. Then run the Streamlit app:**

```cmd
streamlit run app.py
```

Upload any chest X-ray (jpg/jpeg/png) and it will show the predicted class
and a confidence bar chart.

## Why it'll work on the first try

- **No train/inference skew.** `app.py` imports `load_and_preprocess_image()`
  from `preprocessing.py` — the exact same function `train.py` and the
  notebook use internally. There is no second, slightly-different resize/
  normalize implementation living inside the app to accidentally drift out
  of sync.
- **Model save/load is version-tolerant.** `train.py` saves both a native
  `.keras` file and a weights-only backup (`model_utils.save_model_robust`).
  If the `.keras` file ever fails to load (e.g. you trained on one machine
  and deploy the app on another with a different TensorFlow version),
  `app.py` automatically rebuilds the architecture from code and loads
  just the weights instead — no manual intervention needed.
- **Dataset path auto-detection.** `find_dataset_root()` checks a few likely
  folder locations, so you don't get a cryptic `FileNotFoundError` three
  layers deep inside Keras if your working directory isn't exactly what a
  hard-coded path expected.

## Notes on gotchas already fixed during testing

Both of these were caught by actually executing the full pipeline (not
just reading the code), so you won't hit them:

1. **Keras Tuner needs `tensorboard` installed**, even though this project
   never opens a TensorBoard dashboard — it's an internal dependency of
   the tuner's trial-logging code. Already in `requirements.txt`.
2. **`ImageDataGenerator.flow()` + explicit `steps_per_epoch` under Keras 3**
   caused Model 3's augmented training to silently stop consuming batches
   after epoch 1 ("input ran out of data" warning, and val accuracy frozen).
   Fixed by omitting `steps_per_epoch` — the generator already reports its
   own length via `__len__`, so Keras infers it correctly every epoch.
   (See the comment above the `model3.fit(...)` call in `train.py`.)

## Tuning knobs (top of `train.py`)

| Variable | Default | What raising it does |
|---|---|---|
| `EPOCHS` | 15 | More training per model — raise if you have a GPU |
| `TUNER_MAX_TRIALS` | 5 | More Keras Tuner hyperparameter combinations tried |
| `TUNER_EPOCHS` | 6 | More epochs per tuner trial (slower search, better trial signal) |
| `BATCH_SIZE` | 16 | Larger batches train faster on a GPU, may need more RAM |

`IMG_SIZE` (128×128) and `CLASS_NAMES` live in `preprocessing.py` — change
them there once and every script (training, tuning, the app) picks up the
change automatically.

## A note on Model 2/3 (transfer learning) accuracy

`model_utils.py` downloads ImageNet-pretrained MobileNetV2 weights the
first time you run `train.py` (needs internet access once; cached locally
after that). With the full 15 epochs and real pretrained weights, transfer
learning (Models 2-3) should noticeably outperform the from-scratch basic
CNN — that's the expected result for a dataset this small. `train.py`
automatically picks whichever of the (up to) 4 models scores highest on
the test set and saves *that one* for the app, so you don't have to decide
manually.
