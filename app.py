import json
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# ✅ Cached model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('tomato_disease_model.h5')

# ✅ Load class mapping from training (not hardcoded)
@st.cache_resource
def load_class_mapping():
    with open('class_indices.json') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # {0: 'class_name', ...}

model = load_model()
index_to_class = load_class_mapping()

st.title('🌱 Tomato Disease Detector')
st.write('Upload any tomato leaf → Instant diagnosis')

uploaded = st.file_uploader('Choose tomato leaf photo', type=['jpg', 'jpeg', 'png'])

if uploaded:
    # ✅ Force RGB, correct target size
    img = image.load_img(uploaded, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)       # ✅ Correct for MobileNetV2
    img_array = np.expand_dims(img_array, axis=0) # shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction[0])
    predicted_class = index_to_class[predicted_idx]  # ✅ Uses saved mapping
    confidence = np.max(prediction[0]) * 100

    # Display
    st.image(uploaded, caption='Uploaded Leaf', width=300)
    st.success(f"**Diagnosis**: {predicted_class}")
    st.info(f"**Confidence**: {confidence:.1f}%")
    st.progress(confidence / 100)

    # ✅ Bonus: show top 3 predictions for transparency
    st.write("**Top 3 predictions:**")
    top3_idx = np.argsort(prediction[0])[::-1][:3]
    for idx in top3_idx:
        st.write(f"- {index_to_class[idx]}: {prediction[0][idx]*100:.1f}%")

if st.button('ℹ️ About Model'):
    st.write("✅ MobileNetV2 Transfer Learning")
    st.write(f"✅ {len(index_to_class)} Tomato disease classes")
    st.write("✅ ~91% Validation Accuracy")