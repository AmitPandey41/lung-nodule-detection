
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

st.title("Lung Nodule Detection")
st.write("Upload a CT slice/patch image to check for nodule.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_model()

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (64, 64)).astype(np.float32) / 255.0
    img_input = np.expand_dims(np.expand_dims(img_resized, 0), -1)

    prob = float(model.predict(img_input)[0][0])
    label = "Suspicious Nodule" if prob > 0.5 else "Non-Nodule"

    st.image(image, caption="Uploaded Image", width=250)
    st.subheader(f"Prediction: {label}")
    st.write(f"Probability: {prob:.3f}")
