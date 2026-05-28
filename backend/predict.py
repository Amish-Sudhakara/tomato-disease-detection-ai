import json
import numpy as np
from tensorflow.keras.preprocessing import image

from backend.config import IMAGE_SIZE, CLASS_INDICES_PATH
from backend.model_loader import get_model


with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_disease(img_path):
    model = get_model()

    processed_img = preprocess_image(img_path)

    prediction = model.predict(processed_img)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    predicted_class = index_to_class[predicted_index]

    return {
        "class": predicted_class,
        "confidence": round(confidence * 100, 2)
    }
