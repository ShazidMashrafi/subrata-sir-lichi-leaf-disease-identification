#streamlit run app.py

import streamlit as st
import torch
import timm
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time
import psutil
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Litchi Leaf Disease Recognition",
    page_icon="🍃",
    layout="wide"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Anthrax Leaf",
    "Bituminous Leaf",
    "Curl Leaf",
    "Deficiency Leaf",
    "Dry Leaf",
    "Entomosporium Leaf Spot on Woody Ornamentals",
    "Felt Leaf",
    "Fungal Leaf Spot",
    "Healthy Leaf",
    "Leaf Blight",
    "Leaf Gall",
    "Litchi Algal Spot in Non-Direct Sunlight",
    "Litchi Anthracnose on Cloudy Day",
    "Litchi Leaf Mites in Direct Sunlight",
    "Litchi Mayetiola After Raining"
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# CONFIDENCE THRESHOLD
# ============================================================

CONFIDENCE_THRESHOLD = 50  # percent


# ============================================================
# MODEL FILE
# ============================================================

# Resolve model path relative to this script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Create ViT-Tiny architecture
    # --------------------------------------------------------

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Case 1:
    # checkpoint contains state_dict
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "model" in checkpoint and isinstance(
            checkpoint["model"], dict
        ):
            state_dict = checkpoint["model"]

        else:
            # Assume checkpoint itself is state_dict
            state_dict = checkpoint

        # Remove possible prefixes
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):
                new_key = new_key.replace(
                    "module.", "", 1
                )

            if new_key.startswith("model."):
                new_key = new_key.replace(
                    "model.", "", 1
                )

            cleaned_state_dict[new_key] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

    else:
        # Case 2:
        # Entire model was saved using torch.save(model)
        model = checkpoint

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

@st.cache_resource
def get_model_information():

    model = load_model()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    model_size_mb = os.path.getsize(
        MODEL_PATH
    ) / (1024 * 1024)

    return (
        total_parameters,
        trainable_parameters,
        model_size_mb
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image, model):

    image_rgb = image.convert("RGB")

    tensor = transform(image_rgb)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # CPU timing
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():

        output = model(tensor)

        probabilities = F.softmax(
            output,
            dim=1
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    inference_time = (
        end_time - start_time
    ) * 1000

    # Top prediction
    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )

    predicted_index = predicted_index.item()

    confidence = confidence.item()

    # Top 3 predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(3, NUM_CLASSES),
        dim=1
    )

    top_predictions = []

    for prob, idx in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        top_predictions.append({
            "Disease": CLASS_NAMES[idx.item()],
            "Confidence": float(prob.item() * 100)
        })

    # --------------------------------------------------------
    # CONFIDENCE THRESHOLD CHECK
    # --------------------------------------------------------

    predicted_class = CLASS_NAMES[predicted_index]

    if confidence * 100 < CONFIDENCE_THRESHOLD:
        predicted_class = "Unknown"

    return (
        predicted_class,
        confidence * 100,
        inference_time,
        top_predictions
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🍃 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "About",
        "Disease Recognition"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.title(
        "🍃 Litchi Leaf Disease Recognition System"
    )

    st.subheader(
        "Vision Transformer (ViT-Tiny) Based Classification"
    )

    st.write(
        """
        This web application provides on-demand identification
        of litchi leaf diseases using a trained Vision Transformer
        (ViT-Tiny) deep learning model.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Disease Classes",
            "14"
        )

    with col2:
        st.metric(
            "Healthy Class",
            "1"
        )

    with col3:
        st.metric(
            "Input Resolution",
            "224 × 224"
        )

    st.markdown("---")

    st.info(
        """
        Navigate to **Disease Recognition** from the sidebar
        to upload an image or capture a litchi leaf photograph
        using the camera.
        """
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.title("About the System")

    st.write(
        """
        The system was developed to provide an accessible and
        user-friendly interface for litchi leaf disease recognition.
        The application integrates a trained Vision Transformer
        (ViT-Tiny) model with the Streamlit framework.
        """
    )

    st.subheader("Supported Classes")

    for i, disease in enumerate(CLASS_NAMES, start=1):

        st.write(
            f"{i}. {disease}"
        )

    st.markdown("---")

    st.subheader("Model Architecture")

    st.write(
        """
        The backend uses PyTorch and the timm library. The ViT-Tiny
        model receives a 224 × 224 RGB image and produces predictions
        for 15 classes, consisting of 14 disease classes and one
        healthy class.
        """
    )

    st.subheader("Processing Pipeline")

    st.write(
        """
        Image Upload / Camera Capture
        → RGB Conversion
        → Resize to 224 × 224
        → Center Crop
        → Tensor Conversion
        → Normalization
        → ViT-Tiny Inference
        → Softmax Probability
        → Disease Prediction
        """
    )


# ============================================================
# DISEASE RECOGNITION PAGE
# ============================================================

elif page == "Disease Recognition":

    st.title(
        "🔬 Litchi Leaf Disease Recognition"
    )

    st.write(
        """
        Upload a litchi leaf image or capture an image using
        your device camera.
        """
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    try:

        model = load_model()

        (
            total_parameters,
            trainable_parameters,
            model_size_mb
        ) = get_model_information()

    except Exception as e:

        st.error(
            f"Unable to load the model: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # MODEL INFORMATION DISPLAY
    # --------------------------------------------------------

    with st.expander(
        "Model and Computational Information"
    ):

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Model",
                "ViT-Tiny"
            )

        with col2:
            st.metric(
                "Parameters",
                f"{total_parameters / 1e6:.2f} M"
            )

        with col3:
            st.metric(
                "Model Size",
                f"{model_size_mb:.2f} MB"
            )

        with col4:
            st.metric(
                "Device",
                str(DEVICE).upper()
            )

        st.write(
            f"Trainable parameters: "
            f"{trainable_parameters / 1e6:.2f} million"
        )

        st.write(
            f"CPU: {__import__('platform').processor()}"
        )

        memory = psutil.virtual_memory()

        st.write(
            f"System RAM: "
            f"{memory.total / (1024**3):.2f} GB"
        )


    # --------------------------------------------------------
    # INPUT METHOD
    # --------------------------------------------------------

    st.subheader(
        "Select Image Input"
    )

    input_method = st.radio(
        "Choose input method:",
        [
            "Upload Image",
            "Use Camera"
        ],
        horizontal=True
    )

    image = None

    if input_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a litchi leaf image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            )

    else:

        camera_image = st.camera_input(
            "Take a photograph of the litchi leaf"
        )

        if camera_image is not None:

            image = Image.open(
                camera_image
            )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if image is not None:

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Input Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "Prediction"
            )

            if st.button(
                "🔍 Identify Disease",
                type="primary"
            ):

                try:

                    (
                        predicted_class,
                        confidence,
                        inference_time,
                        top_predictions
                    ) = predict_image(
                        image,
                        model
                    )

                    # --------------------------------------------
                    # UNKNOWN CLASS HANDLING
                    # --------------------------------------------

                    if predicted_class == "Unknown":

                        st.warning(
                            f"Prediction: Unknown "
                            f"(Confidence: {confidence:.2f}%)"
                        )

                        st.info(
                            f"The model is not confident "
                            f"enough to classify this image. "
                            f"Confidence is below the "
                            f"{CONFIDENCE_THRESHOLD}% threshold."
                        )

                    else:

                        st.success(
                            f"Prediction: {predicted_class}"
                        )

                    st.info(
                        f"Confidence: "
                        f"{confidence:.2f}%"
                    )

                    st.info(
                        f"Inference time: "
                        f"{inference_time:.2f} ms/image"
                    )

                    # ------------------------------------------------
                    # Confidence interpretation
                    # ------------------------------------------------

                    if predicted_class == "Unknown":

                        st.warning(
                            "Low-confidence prediction. "
                            "Consider capturing another image "
                            "under better lighting and focusing "
                            "on the leaf."
                        )

                    elif confidence >= 90:

                        st.success(
                            "High-confidence prediction"
                        )

                    elif confidence >= 70:

                        st.warning(
                            "Moderate-confidence prediction"
                        )

                    else:

                        st.warning(
                            "Low-confidence prediction. "
                            "Consider capturing another image "
                            "under better lighting and focusing "
                            "on the leaf."
                        )

                    # ------------------------------------------------
                    # TOP 3 PREDICTIONS
                    # ------------------------------------------------

                    st.subheader(
                        "Top-3 Predictions"
                    )

                    results_df = pd.DataFrame(
                        top_predictions
                    )

                    results_df["Confidence"] = (
                        results_df["Confidence"]
                        .round(2)
                        .astype(str)
                        + "%"
                    )

                    st.table(
                        results_df
                    )

                    # ------------------------------------------------
                    # SAVE SESSION RESULT
                    # ------------------------------------------------

                    result_data = pd.DataFrame({
                        "Predicted Class":
                            [predicted_class],

                        "Confidence (%)":
                            [confidence],

                        "Inference Time (ms)":
                            [inference_time],

                        "Device":
                            [str(DEVICE)],

                        "Model":
                            ["ViT-Tiny"]
                    })

                    csv_data = result_data.to_csv(
                        index=False
                    )

                    st.download_button(
                        "Download Prediction Result",
                        data=csv_data,
                        file_name="prediction_result.csv",
                        mime="text/csv"
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Litchi Leaf Disease Recognition System"
)

st.sidebar.caption(
    "Powered by PyTorch + timm + Streamlit"
)
