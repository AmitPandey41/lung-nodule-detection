import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

st.title("Lung Nodule Detection")
st.write("Upload a CT slice/patch image to check for nodule.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cnn_model.h5", compile=False)
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=250)

    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    prediction = model.predict(img, verbose=0)
    probability = float(prediction[0][0])

    if probability > 0.5:
        result = "Suspicious Nodule"
    else:
        result = "Non-Nodule"

    st.subheader("Prediction: " + result)
    st.write("Probability:", round(probability, 3))
