import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Core Directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "..", "temp"))

# MobileNet Paths
MOBILENET_MODEL_PATH = os.path.join(MODELS_DIR, "mobilenet", "tomato_disease_model.keras")
MOBILENET_CLASSES_PATH = os.path.join(MODELS_DIR, "mobilenet", "class_indices.json")

# EfficientNet Paths
EFFICIENTNET_MODEL_PATH = os.path.join(MODELS_DIR, "efficientnet", "new_best_model.keras")
EFFICIENTNET_CLASSES_PATH = os.path.join(MODELS_DIR, "efficientnet", "class_indices.json")

# Shared Properties
TARGET_SIZE = (224, 224)