import streamlit as st
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras import layers, models, backend as K
from PIL import Image
import numpy as np

# Configuration hyperparameters
IMAGE_SIZE = [128, 128]  # Updated to match your image size
SEED = 42
n_hidden_1 = 512
n_hidden_2 = 256
n_hidden_3 = 64
n_hidden_4 = 16
n_hidden_5 = 8
convkernel = (3, 3)  # convolution kernel
poolkernel = (2, 2)  # pooling kernel
dropout_rate = 0.3   # Dropout rate

def get_model():
    K.clear_session()
    
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        strategy = tf.distribute.get_strategy()  # default strategy (CPU/GPU)
    
    with strategy.scope():
        inp1 = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
        
        # Encoder
        x = tf.keras.layers.Conv2D(n_hidden_1, convkernel, padding='same')205(inp1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(n_hidden_1, convkernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        x = tf.keras.layers.Conv2D(n_hidden_2, convkernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(n_hidden_2, convkernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        x = tf.keras.layers.Conv2D(n_hidden_3, convkernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(n_hidden_3, convkernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        x = tf.keras.layers.Conv2D(n_hidden_4, convkernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(n_hidden_4, convkernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        x = tf.keras.layers.Conv2D(n_hidden_5, convkernel, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        encoded = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(x)
        
        # Decoder
        x = tf.keras.layers.Conv2DTranspose(n_hidden_5, convkernel, strides=2, activation='relu', padding='same')(encoded)
        x = tf.keras.layers.Conv2DTranspose(n_hidden_4, convkernel, strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(n_hidden_3, convkernel, strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(n_hidden_2, convkernel, strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2DTranspose(n_hidden_1, convkernel, strides=2, activation='relu', padding='same')(x)
        
        decoded = tf.keras.layers.Conv2DTranspose(3, convkernel, activation="sigmoid", padding='same')(x)
        
        # Define the model
        model = tf.keras.models.Model(inputs=inp1, outputs=decoded)
        
        # Compile the model
        # opt = tfa.optimizers.RectifiedAdam(lr=3e-4)
        # model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        
        # Use the Adam optimizer with the same learning rate
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)

        # Compile the model with the Adam optimizer
        model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def preprocess_image(image, target_size=(128, 128)):
    """
    Loads an image, resizes it, and normalizes pixel values.
    
    Parameters:
    - image_path (str): Path to the image file.
    - target_size (tuple): Desired image size (width, height).
    
    Returns:
    - np.array: Preprocessed image array.
    """
    # Load the image
    # image = Image.open(image_path).convert('RGB')  # Ensure it's RGB
    # Resize the image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    # Convert to NumPy array
    img_array = np.array(image)
    # Normalize to [0, 1]
    img_array_normalized = img_array / 255.0
    return img_array_normalized

def classify_image(model, image_path, threshold):
    """
    Classifies the input image as "Normal" or "Abnormal" based on the reconstruction error.
    
    Parameters:
    - model: Trained autoencoder model.
    - image_path (str): Path to the input image.
    - threshold (float): The threshold for classifying an image as normal or abnormal.
    
    Returns:
    - str: "Normal" if the image is classified as normal, "Abnormal" if classified as abnormal.
    - float: The reconstruction error (MSE).
    """
    # Step 1: Preprocess the image
    new_image = preprocess_image(image_path)
    
    # Step 2: Add batch dimension
    new_image_batch = np.expand_dims(new_image, axis=0)
    
    # Step 3: Predict the reconstructed image
    reconstructed_image_batch = model.predict(new_image_batch)
    reconstructed_image = reconstructed_image_batch[0]
    
    # Step 4: Calculate the reconstruction error (MSE)
    mse = np.mean(np.power(new_image - reconstructed_image, 2))
    
    # Step 5: Classify the image based on the threshold
    if mse > threshold:
        return "Abnormal", mse
    else:
        return "Normal", mse

# Instantiate the model
model = get_model()

# Load the saved weights
model.load_weights('models/anomaly_detection_model.h5')

threshold = 0.0431467411108315

st.title("🔎 Anomaly Detection")

# Upload one or more images
uploaded_files = st.file_uploader("Choose images to upload", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
        # Classify the image
        classification, reconstruction_error = classify_image(model, image, threshold)

        # Display the prediction result
        st.write(f"Prediction Result: {classification}")
        st.write(f"Reconstruction Error (MSE): {reconstruction_error}")
        st.markdown("<hr>", unsafe_allow_html=True)