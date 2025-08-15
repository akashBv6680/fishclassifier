# ==============================================================================
# Streamlit App for Fish Classification with VGG16 (11 Classes)
# This script is a simple web app to classify fish images using a
# pre-trained VGG16 model with an updated list of 11 class names.
# ==============================================================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Set the title and header of the Streamlit app
st.set_page_config(page_title="Fish Classifier (11 Classes)", layout="centered")
st.title("🐟 Fish Species Classifier")
st.markdown("### Predicting 11 different fish classes")
st.write("Upload an image of a fish, and the model will predict its species.")

# ==============================================================================
# --- 1. MODEL LOADING AND CLASS DEFINITION ---
# ==============================================================================
# Define the path to your saved model file.
# NOTE: The model file `best_model_VGG16.h5` must be in the same directory
# as this script on your GitHub repository.
MODEL_PATH = "best_model_VGG16.h5"

# --- IMPORTANT: USE THE 11 CLASS NAMES YOU PROVIDED ---
# These class names must be in the exact same order as they were during training.
CLASS_NAMES = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]

# Load the trained model, with a try-except block for robustness.
try:
    @st.cache_resource
    def load_the_model():
        """Caches the model loading to prevent reloading on every interaction."""
        return load_model(MODEL_PATH)

    model = load_the_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model. Please make sure '{MODEL_PATH}' is in your GitHub repository. Error: {e}")
    st.stop() # Stop the app if the model can't be loaded

# ==============================================================================
# --- 2. IMAGE UPLOADER AND PREDICTION LOGIC ---
# ==============================================================================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image to match the model's expected input size
    # VGG16 expects a 224x224 input
    image_resized = image.resize((224, 224))
    
    # Convert the image to a numpy array and scale it
    image_array = np.array(image_resized) / 255.0  # Scale pixel values to [0, 1]
    
    # Expand dimensions to create a batch of size 1
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction using the model
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions)

    # Display the results to the user
    st.success(f"**Prediction:** {predicted_class_name}")
    st.info(f"**Confidence:** {confidence:.2f}")

    # Display a bar chart of the top 3 predictions
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
