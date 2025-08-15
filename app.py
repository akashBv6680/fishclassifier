import streamlit as st
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

# --- Page Configuration ---
st.set_page_config(
    page_title="Hugging Face Fish Species Classifier",
    page_icon="🐠",
    layout="centered"
)

# --- App Title and Introduction ---
st.title("🐠 Hugging Face Fish Species Classifier")
st.write("Upload an image of a fish to get a prediction. This app uses a pre-trained Vision Transformer model from Hugging Face.")
st.markdown("---")

# --- Model Loading ---
# We use st.cache_resource to cache the model so it only downloads once
# and doesn't reload on every user interaction. This is crucial for performance.
@st.cache_resource
def load_model():
    """
    Loads the pre-trained Vision Transformer model and feature extractor from Hugging Face.
    The model is 'google/vit-base-patch16-224'.
    """
    # The feature extractor preprocesses the image to the model's required format
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    # The model itself
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    return feature_extractor, model

# Load the model and feature extractor
feature_extractor, model = load_model()

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a fish image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to classify the fish species."
)

# --- Prediction Logic ---
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.subheader("Classifying...")

    # A simple loading spinner to show the user that work is being done
    with st.spinner('Making a prediction...'):
        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Get the model predictions
        outputs = model(**inputs)
        logits = outputs.logits

        # Get the predicted class and its score
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_label = model.config.id2label[predicted_class_idx]
        predicted_score = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class_idx].item()

        # Display the results
        st.success("Classification complete!")
        st.metric(
            label="Predicted Class",
            value=f"**{predicted_class_label}**"
        )
        st.info(f"Confidence: {predicted_score:.2f}")

    st.markdown("---")
    st.write(
        "**Note:** This is a general-purpose model. For better accuracy on your specific fish classes, "
        "you would need to fine-tune this model on your own dataset of 11 fish species."
    )

