import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐶",
    layout="centered"
)

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_cat_model.keras")

model = load_model()

# App title
st.title("🐶 Dog vs Cat Classification 🐱")

st.write("Upload an image to predict whether it is a Dog or Cat.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", width=300)

    # Resize image
    img = image.resize((64, 64))

    # Convert image into array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    if st.button("Predict"):

        prediction = model.predict(img_array, verbose=0)

        # Assumes training class order: Cat = 0, Dog = 1
        if prediction[0][0] > 0.5:
            st.success("🐶 Prediction: Dog")
        else:
            st.success("🐱 Prediction: Cat")
