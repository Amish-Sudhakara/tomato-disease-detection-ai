import io
import requests
import gradio as gr
from PIL import Image

API_BASE_URL = "http://127.0.0.1:8000/predict"

def send_image_to_backend(pil_image, model_choice):
    if pil_image is None:
        return None

    # Map radio option text to backend router names
    model_type = "mobilenet" if "MobileNet" in model_choice else "efficientnet"
    api_url = f"{API_BASE_URL}/{model_type}"

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    try:
        files = {"file": ("sample.jpg", buffer, "image/jpeg")}
        response = requests.post(api_url, files=files, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            return {f"⚠️ Error: Server returned status {response.status_code}": 1.0}

    except requests.exceptions.RequestException as error:
        return {f"❌ Failed to reach API. Is backend running?\n{str(error)}": 1.0}

# ==========================================
# GRADIO UI DESIGN (MobileNet Theme Base)
# ==========================================

with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="green",
        font=["Exo 2", "sans-serif"]
    )
) as demo:

    gr.Markdown("""
    # 🍅 AI Tomato Disease Detector
    
    Professional-grade deep learning diagnostic utility powered by dual-core architectures.
    
    ⚡ *Select your architecture below to process image diagnostics via the unified API backend.*
    """)

    with gr.Row():

        with gr.Column(scale=1):
            input_img = gr.Image(
                type="pil",
                label="Upload Leaf Sample"
            )
            
            model_selector = gr.Radio(
                choices=["MobileNetV2 Core (Fast & Lightweight)", "EfficientNetB0 Core (High Accuracy Fine-Tuned)"],
                value="MobileNetV2 Core (Fast & Lightweight)",
                label="Select AI Architecture Core"
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

        with gr.Column(scale=1):
            output_labels = gr.Label(
                num_top_classes=3,
                label="Probability Distribution Analysis"
            )

    submit_btn.click(
        fn=send_image_to_backend,
        inputs=[input_img, model_selector],
        outputs=output_labels
    )

    clear_btn.click(
        fn=lambda: (None, "MobileNetV2 Core (Fast & Lightweight)", None),
        inputs=None,
        outputs=[input_img, model_selector, output_labels]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)