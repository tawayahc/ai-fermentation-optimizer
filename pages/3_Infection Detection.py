import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import tempfile

# Streamlit interface
st.title("🚨 Infection Detection")

# Upload one or more images
uploaded_files = st.file_uploader("Choose images to upload", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    rf = Roboflow(api_key="ZxZaHiLs1eGAWpKffxJ1")
    project = rf.workspace().project("yeast-cells-detection")
    model = project.version(1).model

    legend_html = """
        <div class="legend">
            <div class="legend-item">
                <div class="color-box yellow"></div>
                <span class="label">Flocculation Infection</span>
            </div>
            <div class="legend-item">
                <div class="color-box purple"></div>
                <span class="label">Type L Infection</span>
            </div>
        </div>
        <style>
            .legend {
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #F0F2F6;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }

            .legend-item {
                display: flex;
                align-items: center;
                margin-right: 20px;
            }

            .color-box {
                width: 24px;
                height: 24px;
                border-radius: 6px;
                margin-right: 10px;
            }

            .yellow {
                background-color: #ffce21;
            }

            .purple {
                background-color: #8622FF;
            }

            .label {
                font-family: 'Arial', sans-serif;
                font-size: 15.5px;
                color: #31333F;
            }
        </style>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    for uploaded_file in uploaded_files:
        # Open the uploaded image with PIL
        image = Image.open(uploaded_file)

        # Display the uploaded image
        # st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
            image.save(temp_image_file, format='JPEG')
            temp_image_path = temp_image_file.name  # Get the temp file path

        # Send the temp file path to the Roboflow model for prediction
        prediction = model.predict(temp_image_path, confidence=40, overlap=30).json()

        # Display the prediction result
        # st.write(prediction)

        # Check if 'predictions' or a similar key exists in the response
        if 'predictions' in prediction:
            detections = prediction['predictions']  # Get the list of detections

            # Load the image again for drawing the bounding boxes
            image_with_boxes = Image.open(temp_image_path)
            draw = ImageDraw.Draw(image_with_boxes)

            # Set the font and size for the label
            try:
                font = ImageFont.truetype("Gidole-Regular.ttf", size=25)  # Use Arial font with size 24
            except IOError:
                font = ImageFont.load_default()  # Fallback to default font if Arial is unavailable

            # Draw rectangles based on the predictions
            for obj in detections:
                x = obj["x"]
                y = obj["y"]
                width = obj["width"]
                height = obj["height"]

                # Calculate the box coordinates
                top_left = (x - width // 2, y - height // 2)
                bottom_right = (x + width // 2, y + height // 2)

                if obj['class'] == 'Flocculation':
                    color = "#ffce21"
                else:
                    color = "#8622FF"

                # Draw the bounding box
                draw.rectangle([top_left, bottom_right], outline=color, width=3)

                # Add class label and confidence with increased font size
                label = f"{obj['class']} ({obj['confidence']:.2f})"
                # draw.text((x - width // 2, y - height // 2 - 30), label, fill=color, font=font)

            # Display the image with bounding boxes
            st.image(image_with_boxes, caption="", use_column_width=True)

        else:
            st.error("No predictions found in the response.")

    st.markdown("<hr>", unsafe_allow_html=True)
