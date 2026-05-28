import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

print("🚀 Tomato Disease Detector - Production Ready")

# =========================
# DATA PIPELINE
# =========================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    './data/plantvillage/',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    './data/plantvillage/',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)

print(f"✅ Training: {train_gen.samples} images")
print(f"✅ Number of classes: {num_classes}")

# =========================
# MODEL
# =========================

base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled!")

# =========================
# TRAINING
# =========================

print("🎯 Training starts...")

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        )
    ]
)

# =========================
# PLOTS
# =========================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("training_plots.png")

plt.show()

print("📊 training_plots.png saved!")

# =========================
# SAVE MODELS
# =========================

# Main production model
model.save(
    "../backend/models/tomato_disease_model.keras"
)

# Backup model
model.save(
    "backup_models/tomato_disease_model.h5"
)

print("✅ Models saved successfully!")

# =========================
# FINAL RESULT
# =========================

final_acc = max(history.history['val_accuracy'])

print(f"🎉 Final Validation Accuracy: {final_acc:.3f}")