# ==============================================================================
# Streamlit App with Model Download from Google Drive
# This script is now configured to download your specific model file
# from the Google Drive link you provided.
# ==============================================================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown  # gdown is used to download from Google Drive

st.set_page_config(page_title="Fish Classifier (11 Classes)", layout="centered")
st.title("🐟 Fish Species Classifier")
st.markdown("### Predicting 11 different fish classes")
st.write("This app downloads the model from Google Drive during deployment. Please be patient!")

# ==============================================================================
# --- 1. MODEL DOWNLOAD AND LOADING ---
# ==============================================================================
# Define the Google Drive file ID for your model.
# The ID '1cToYesYVhshDJdDgAyERI-kxw7LAvPpM' was extracted from your link.
MODEL_DRIVE_ID = '1cToYesYVhshDJdDgAyERI-kxw7LAvPpM'
MODEL_PATH = "best_model_VGG16.h5"

# --- IMPORTANT: USE THE 11 CLASS NAMES YOU PROVIDED ---
CLASS_NAMES = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]

# A function to download the model if it doesn't exist.
@st.cache_resource
def load_the_model():
    """
    Downloads the model from Google Drive if it doesn't exist locally,
    then loads and caches it.
    """
    if not os.path.exists(MODEL_PATH):
        st.info(f"Downloading model from Google Drive... This may take a moment.")
        try:
            gdown.download(f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}', MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading the model. Please check the Google Drive ID and sharing permissions. Error: {e}")
            st.stop()

    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading the model. Error: {e}")
        st.stop()

model = load_the_model()

# ==============================================================================
# --- 2. IMAGE UPLOADER AND PREDICTION LOGIC ---
# ==============================================================================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions)

    st.success(f"**Prediction:** {predicted_class_name}")
    st.info(f"**Confidence:** {confidence:.2f}")

    st.write("---")
    st.subheader("Top Predictions")
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_confidences = [predictions[0][i] for i in top_3_indices]
    top_3_classes = [CLASS_NAMES[i] for i in top_3_indices]
    
    chart_data = {
        'Species': top_3_classes,
        'Confidence': top_3_confidences
    }
    st.bar_chart(chart_data, x='Species', y='Confidence')
