import streamlit as st
from PIL import Image

st.title("🔎 Anomaly Detection")

# Allow users to upload one or more images
uploaded_files = st.file_uploader("Choose images to upload", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
        # Display the prediction result
        st.write(f"Prediction Result!")