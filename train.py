import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

print("🚀 Tomato Disease Detector - Production Ready")

# 1. Data pipeline (MobileNetV2 optimized preprocessing)
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
print(f"✅ Training: {train_gen.samples} images, {num_classes} classes")
print(f"✅ Classes: {list(train_gen.class_indices.keys())}")

# 2. MobileNetV2 Transfer Learning
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Model compiled!")

# 3. Training
print("🎯 Training starts... (grab chai ☕)")
history = model.fit(
    train_gen, epochs=10,
    validation_data=val_gen,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# 4. Diagnostic Plots (ML Engineering Standard)
plt.figure(figsize=(12, 4))

# Accuracy plot (detect overfitting)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Acc')
plt.plot(history.history['val_accuracy'], label='Validation Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot (detect underfitting/convergence)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plots.png')
plt.show()
print("📊 Plots saved as 'training_plots.png' - perfect for reports!")

# 5. Results + Save
final_acc = max(history.history['val_accuracy'])
print(f"🎉 Training complete! Final val accuracy: {final_acc:.3f}")
model.save('tomato_disease_model.h5')
print("✅ Model saved! Next: streamlit run app.py")