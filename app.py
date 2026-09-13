import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

st.title("Lung Nodule Detection")
st.write("Upload a CT slice/patch image to check for nodule.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.h5", compile=False)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload CT image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)

    st.image(image, caption="Uploaded Image", width=250)

    # Resize image to the size used by the model
    img = cv2.resize(img, (64, 64))

    # Convert image to float
    img = img.astype(np.float32)

    # Normalize image
    img = img / 255.0

    # Add channel and batch dimensions
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)
    probability = float(prediction[0][0])

    if probability >= 0.5:
        result = "Suspicious Nodule"
    else:
        result = "Non-Nodule"

    st.subheader("Prediction: " + result)
    st.write("Probability:", round(probability, 3))
