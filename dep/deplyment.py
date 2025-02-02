import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import torchvision.models as models

# Optional: custom CSS styling
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .main-container {
        animation: fadeIn 1.5s ease-in-out;
        text-align: center;
        padding: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #ffa502);
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        transition: 0.3s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load the trained ResNet18 model and class names
@st.cache_resource
def load_model():
    model_path = "covid_xray_model.pth"
    if not os.path.exists(model_path):
        st.error("Error: covid_xray_model.pth not found! Please upload the trained model file.")
        return None, None

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    # Infer number of classes
    num_classes = state_dict['fc.weight'].shape[0]

    # Define custom class labels
    if num_classes == 2:
        class_names = ["COVID-19 Positive", "COVID-19 Negative"]
    elif num_classes == 3:
        class_names = ["COVID-19 Positive", "COVID-19 Negative", "Other Disease"]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]  # Fallback

    # Initialize ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load model weights
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names

# Load model
model, class_names = load_model()

# Streamlit UI layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🚑 COVID-19 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to classify.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Ensure image is in RGB (ResNet expects 3-channel input)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    with st.spinner("Analyzing image... Please wait 🚀"):
        # Preprocess image
        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probabilities).item()
            pred_class = class_names[pred_idx]

        # Prepare probability dictionary for display
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write("**Probabilities:**", prob_dict)

st.write("<br><br>**Note:** This is a demo and not a substitute for professional medical advice.",
         unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
