import json
import os
import tensorflow as tf
from backend.config import (
    MOBILENET_MODEL_PATH, MOBILENET_CLASSES_PATH,
    EFFICIENTNET_MODEL_PATH, EFFICIENTNET_CLASSES_PATH
)

MODELS = {
    "mobilenet": None,
    "efficientnet": None
}

CLASSES = {
    "mobilenet": {},
    "efficientnet": {}
}

def build_efficientnet_architecture(num_classes: int):
    """Rebuilds the Sequential wrapper architecture for EfficientNet weights binding."""
    base_model = tf.keras.applications.EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def load_all_models():
    global MODELS, CLASSES
    
    # ---- Load MobileNet ----
    if MODELS["mobilenet"] is None:
        print("🚀 Preloading MobileNetV2 Architecture...")
        if os.path.exists(MOBILENET_CLASSES_PATH):
            with open(MOBILENET_CLASSES_PATH, "r") as f:
                mobilenet_idx = json.load(f)
            CLASSES["mobilenet"] = {int(v): k for k, v in mobilenet_idx.items()}
        
        if os.path.exists(MOBILENET_MODEL_PATH):
            MODELS["mobilenet"] = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
            print("✅ MobileNetV2 loaded successfully!")
        else:
            print(f"⚠️ MobileNet file missing at: {MOBILENET_MODEL_PATH}")

    # ---- Load EfficientNet ----
    if MODELS["efficientnet"] is None:
        print("🚀 Preloading EfficientNetB0 via Architecture Reconstruction...")
        if os.path.exists(EFFICIENTNET_CLASSES_PATH):
            with open(EFFICIENTNET_CLASSES_PATH, "r") as f:
                effnet_idx = json.load(f)
        else:
            effnet_idx = {f"Class_{i}": i for i in range(10)}
            
        CLASSES["efficientnet"] = {int(v): k for k, v in effnet_idx.items()}
        num_classes = len(effnet_idx)
        
        if os.path.exists(EFFICIENTNET_MODEL_PATH):
            try:
                eff_model = build_efficientnet_architecture(num_classes)
                eff_model.load_weights(EFFICIENTNET_MODEL_PATH)
                MODELS["efficientnet"] = eff_model
                print("✅ EfficientNetB0 loaded and bound successfully!")
            except Exception as e:
                print(f"❌ EfficientNet structural binding error: {str(e)}")
        else:
            print(f"⚠️ EfficientNet file missing at: {EFFICIENTNET_MODEL_PATH}")

def get_model(model_name: str):
    if MODELS[model_name] is None:
        load_all_models()
    return MODELS[model_name]

def get_class_mappings(model_name: str):
    if not CLASSES[model_name]:
        load_all_models()
    return CLASSES[model_name]