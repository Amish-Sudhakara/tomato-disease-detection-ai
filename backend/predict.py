import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from backend.model_loader import get_model, get_class_mappings
from backend.config import TARGET_SIZE

def clean_disease_name(name: str) -> str:
    """
    Standardizes inconsistent naming conventions between datasets dynamically.
    Transforms formats like 'Tomato___Early_Blight' or 'Blight' into 'Tomato Early Blight'.
    """
    # Replace multiple or single underscores with spaces
    clean = name.replace('___', ' ').replace('_', ' ')
    
    # Ensure consistency if one dataset skips the 'Tomato' prefix
    if not clean.lower().startswith("tomato"):
        clean = f"Tomato {clean}"
        
    # Standardize to title case (e.g., "Tomato Early Blight")
    return clean.strip().title()

def predict_disease(image_path: str, model_type: str) -> dict:
    """Loads image, runs deep learning inference using specified model, and returns probabilities."""
    model = get_model(model_type)
    index_to_class = get_class_mappings(model_type)
    
    if model is None:
        return {"Error": f"Model core '{model_type}' is uninitialized or missing."}

    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_resized = img.resize(TARGET_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)

    # Apply specific model scaling rules
    if model_type == "mobilenet":
        img_array = mobilenet_preprocess(img_array)
    elif model_type == "efficientnet":
        img_array = efficientnet_preprocess(img_array)
        
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0]

    results = {}
    for idx, confidence in enumerate(prediction):
        class_name = index_to_class.get(idx, f"Unknown Class {idx}")
        
        # Unify name string mapping rules before passing to backend response dictionary
        clean_name = clean_disease_name(class_name)
        
        results[clean_name] = float(confidence)

    return results