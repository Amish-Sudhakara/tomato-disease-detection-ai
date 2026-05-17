import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("✅ Script started")

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
train_gen = datagen.flow_from_directory(
    'C:/Users/user/Desktop/Tomato Model/plant-disease/data/plantvillage/',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

with open('class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

print("✅ Done!", train_gen.class_indices)