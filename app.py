"""
app.py
======
Streamlit app: upload a chest X-ray, get a COVID-19 / Normal / Viral
Pneumonia prediction from the model train.py (or the notebook) produced.

Run with:
    streamlit run app.py

Design choices worth knowing:
- This file imports load_and_preprocess_image from preprocessing.py rather
  than writing its own resize/scale code. That is the single most
  important line in this file: it's what guarantees the image the model
  sees here is prepared IDENTICALLY to how training images were prepared,
  so the model's reported test accuracy is actually representative of
  what you'll see in the app.
- The model is loaded once and cached (@st.cache_resource) instead of on
  every user interaction, so the app stays responsive after the first
  (slower) load.
"""

import os
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocessing import load_and_preprocess_image, CLASS_NAMES

MODEL_DIR = "models"
MODEL_NAME = "best_model.tflite"

st.set_page_config(page_title="COVID-19 X-ray Classifier", page_icon="🫁", layout="centered")

@st.cache_resource(show_spinner="Loading TFLite model...")
def get_interpreter_and_metadata():
    """
    Load the TFLite model and metadata.json once per app session.
    """
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        return None, None

    with open(metadata_path) as f:
        metadata = json.load(f)

    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        return None, metadata

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter, metadata


def run_inference(interpreter, img_array):
    """
    Run inference on a preprocessed image using the TFLite interpreter.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure float32 input
    img_array = img_array.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]  # probabilities vector


def main():
    st.title("🫁 COVID-19 Chest X-ray Classifier")
    st.write(
        "Upload a chest X-ray image and this app will classify it as "
        "**COVID-19**, **Normal**, or **Viral Pneumonia** using a CNN "
        "trained on the Kaggle COVID-19 X-ray dataset."
    )
    st.warning(
        "⚠️ This is a learning/portfolio project, NOT a medical diagnostic "
        "tool. Do not use it to make real healthcare decisions.",
        icon="⚠️",
    )

    interpreter, metadata = get_interpreter_and_metadata()
    if interpreter is None:
        st.error(
            f"No trained model found in `{MODEL_DIR}/`. "
            f"Run `python train.py` (or the notebook end-to-end) first "
            f"to create `{MODEL_DIR}/{MODEL_NAME}`."
        )
        return

    with st.expander("ℹ️ About the loaded model"):
        st.write(f"**Architecture:** {metadata['best_model_name']}")
        st.write(f"**Test accuracy:** {metadata['test_accuracy']:.2%}")
        st.write(f"**Input image size:** {metadata['img_size'][0]}x{metadata['img_size'][1]}")

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)

        # Identical preprocessing to training
        uploaded_file.seek(0)
        img_array = load_and_preprocess_image(uploaded_file)

        with st.spinner("Classifying..."):
            probabilities = run_inference(interpreter, img_array)

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        with col2:
            st.subheader("Prediction")
            if predicted_class == "Covid":
                st.error(f"**{predicted_class}** ({confidence:.1%} confidence)")
            elif predicted_class == "Viral Pneumonia":
                st.warning(f"**{predicted_class}** ({confidence:.1%} confidence)")
            else:
                st.success(f"**{predicted_class}** ({confidence:.1%} confidence)")

        st.subheader("Confidence by class")
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.barh(CLASS_NAMES, probabilities * 100,
                       color=["#d62728", "#2ca02c", "#ff7f0e"])
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        for bar, prob in zip(bars, probabilities):
            ax.text(prob * 100 + 1, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.1%}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    
