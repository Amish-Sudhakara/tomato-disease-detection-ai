from tensorflow.keras.models import load_model
from backend.config import MODEL_PATH

model = None


def get_model():
    global model

    if model is None:
        print("Loading model...")
        model = load_model(MODEL_PATH)

    return model
