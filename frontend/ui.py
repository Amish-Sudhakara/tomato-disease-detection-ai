import json
import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("🚀 Loading AI model... Please wait")

# =========================
# PRELOAD MODEL AT STARTUP
# =========================

MODEL = tf.keras.models.load_model(
    'backend/models/tomato_disease_model.h5'
)

with open('backend/models/class_indices.json') as f:
    class_indices = json.load(f)

INDEX_TO_CLASS = {
    v: k for k, v in class_indices.items()
}

print("✅ AI model loaded successfully!")

# =========================
# PREDICTION FUNCTION
# =========================

def predict_tomato_disease(pil_image):

    if pil_image is None:
        return None

    # Resize image
    img_resized = pil_image.resize((224, 224))

    # Convert to numpy
    img_array = np.array(
        img_resized,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = MODEL.predict(
        img_array,
        verbose=0
    )[0]

    # Format output
    results = {}

    for idx, confidence in enumerate(prediction):

        class_name = INDEX_TO_CLASS[idx]

        class_name = (
            class_name
            .replace('_', ' ')
            .title()
        )

        results[class_name] = float(confidence)

    return results

# =========================
# GRADIO UI
# =========================

with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="green",
        font=["Exo 2", "sans-serif"]
    )
) as demo:

    gr.Markdown("""
    # 🍅 AI Tomato Disease Detector
    
    Professional-grade deep learning diagnostic utility powered by MobileNetV2.
    
    ⚡ AI model preloaded for instant predictions.
    """)

    with gr.Row():

        # LEFT PANEL
        with gr.Column(scale=1):

            input_img = gr.Image(
                type="pil",
                label="Upload Leaf Sample"
            )

            with gr.Row():

                clear_btn = gr.Button(
                    "🔄 Reset Dashboard",
                    variant="secondary"
                )

                submit_btn = gr.Button(
                    "🔬 Analyse Tissue",
                    variant="primary"
                )

        # RIGHT PANEL
        with gr.Column(scale=1):

            output_labels = gr.Label(
                num_top_classes=3,
                label="Probability Distribution Analysis"
            )

    # =========================
    # BUTTON ACTIONS
    # =========================

    submit_btn.click(
        fn=predict_tomato_disease,
        inputs=input_img,
        outputs=output_labels
    )

    clear_btn.click(
        fn=lambda: (None, None),
        inputs=None,
        outputs=[input_img, output_labels]
    )

# =========================
# LAUNCH APP
# =========================

if __name__ == "__main__":

    demo.launch()