import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained MNIST model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_mnist_model.h5')

model = load_model()

st.title('MNIST Digit Recognizer')

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 20)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", False)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Set a confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # You may need to adjust this value

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    # Convert the image to grayscale
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
    img = img.convert('L')
    
    # Resize image to 28x28 as required by the MNIST model
    img = img.resize((28, 28))
    
    # Convert image to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255
    
    # Expand dimensions to match the input shape of the model (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    # Predict the digit using the model
    prediction = model.predict(img_array)[0]
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Display the prediction and confidence
    if confidence >= CONFIDENCE_THRESHOLD:
        st.write(f'Predicted Digit: {predicted_digit}')
        st.write(f'Confidence: {confidence:.2f}')
    else:
        st.write("The input doesn't seem to be a clear digit. Please try drawing again.")

# Add instructions
st.write('Draw a digit on the canvas above and the model will try to recognize it.')