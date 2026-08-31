# --- Add to model_utils.py (or use standalone) -----------------------------
# Loading and predicting with the int8-quantized TFLite model.
# Swap this in wherever app.py currently calls model.predict(img_array).

import tensorflow as tf
import numpy as np


def load_tflite_model(tflite_path="best_model_int8.tflite"):
    """Load the quantized model once (cache this in Streamlit with
    @st.cache_resource, same as the original load_model_robust call)."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(interpreter, img_array):
    """
    Drop-in replacement for `model.predict(img_array)`.
    img_array must already be preprocessed by preprocessing.load_and_preprocess_image
    (shape (1, 128, 128, 3), float32, scaled to [0,1]) - identical preprocessing,
    only the prediction call changes.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img_array.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]  # same shape as before: (3,) probabilities


# --- Usage in app.py, replacing the two current lines -----------------------
#   model = load_model_robust(...)              -->  interpreter = load_tflite_model()
#   probabilities = model.predict(img_array)[0]  -->  probabilities = predict_tflite(interpreter, img_array)
# Everything downstream (argmax, confidence bar chart) stays exactly the same.
