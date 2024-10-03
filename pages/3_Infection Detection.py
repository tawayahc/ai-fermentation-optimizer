import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from utils.mp_hack_imageprocessing import CellClassifierCNN  # Assuming this is the model class

# Load the model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CellClassifierCNN().to(device)
    model.load_state_dict(torch.load('models/cell_classification_model.pth', map_location=device))  # Update the path to the model file
    model.eval()
    return model, device

# Image preprocessing function
def preprocess_image(image):
    data_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    return image

# Inference function
def predict(model, device, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()

# Streamlit interface
st.title("🚨 Infection Detection")

# Load the model once
model, device = load_model()

# Upload one or more images
uploaded_files = st.file_uploader("Choose images to upload", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
        # Preprocess the image
        image_tensor = preprocess_image(image)
        
        # Get prediction from the model
        prediction = predict(model, device, image_tensor)

        # Display the prediction result
        if prediction[0] == 0:
            st.write("Prediction Result: Normal")
        elif prediction[0] == 1:
            st.write("Prediction Result: Infection [Flocculation]")
        else:
            st.write("Prediction Result: Infection [Type L]")

        st.markdown("<hr>", unsafe_allow_html=True)
