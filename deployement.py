import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('brain.h5')  # Replace with your model's path

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape
    image = cv2.resize(image, (256, 256))  # Adjust size according to your model's input shape
    # Normalize the image
    image = image.astype('float32') / 256
    # Expand dimensions to match the model input
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app code
st.title("Brain Tumor Classification")
st.write("Upload an MRI image to classify whether it is a brain tumor or not.")

# Uploading the image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    # Convert image to numpy array
    image = np.array(image)

    # Preprocess the image for the model
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)

    # Display prediction result
    if prediction[0][0] < 0.5:  # Adjust threshold based on your model
        st.write("The model predicts: *Brain Tumor Detected")
    else:
        st.write("The model predicts: **No Brain Tumor*")