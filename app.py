import json
import numpy as np
from PIL import Image
import gradio as gr

# Global variables to cache the model so it only loads ONCE when analyzed
MODEL = None
INDEX_TO_CLASS = None

def predict_tomato_disease(pil_image):
    global MODEL, INDEX_TO_CLASS
    
    if pil_image is None:
        return None
    
    # ── LAZY LOADING TENSORFLOW ──
    # Avoids sluggish app loading times by only importing when an image is sent
    if MODEL is None:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        
        MODEL = tf.keras.models.load_model('tomato_disease_model.h5')
        with open('class_indices.json') as f:
            class_indices = json.load(f)
        INDEX_TO_CLASS = {v: k for k, v in class_indices.items()}
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # ── AI PREPROCESSING & INFERENCE ──
    img_resized = pil_image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = MODEL.predict(img_array, verbose=0)[0]
    
    # Format cleanly for Gradio's minimalist dashboard bars
    results = {}
    for idx, confidence in enumerate(prediction):
        class_name = INDEX_TO_CLASS[idx].replace('_', ' ').title()
        results[class_name] = float(confidence)
        
    return results

# ── MINIMAL, SCIENTIFIC SAAS UI ──
# Changed gr.themes.Dark to gr.themes.Default to fix the AttributeError
with gr.Blocks(theme=gr.themes.Default(primary_hue="green", font=["Exo 2", "sans-serif"])) as demo:
    gr.Markdown("""
    # 🍅 AI Tomato Disease Detector
    *Professional-grade deep learning diagnostic utility powered by MobileNetV2.*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Native clipboard pasting and drag-and-drop support out of the box
            input_img = gr.Image(type="pil", label="Upload Leaf Sample (Drag & Drop or Ctrl+V to Paste)")
            with gr.Row():
                clear_btn = gr.Button("🔄 Reset Dashboard", variant="secondary")
                submit_btn = gr.Button("🔬 Analyse Tissue", variant="primary")
                
        with gr.Column(scale=1):
            output_labels = gr.Label(num_top_classes=3, label="Probability Distribution Analysis")

    # Bind actions
    submit_btn.click(fn=predict_tomato_disease, inputs=input_img, outputs=output_labels)
    clear_btn.click(fn=lambda: (None, None), inputs=None, outputs=[input_img, output_labels])

if __name__ == "__main__":
    demo.launch()