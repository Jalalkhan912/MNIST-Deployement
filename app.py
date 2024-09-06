from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the trained MNIST model
model = tf.keras.models.load_model('my_mnist_model.h5')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get image from the request
    data = request.get_json()

    # Decode base64 image
    img_data = base64.b64decode(data['image'].split(',')[1])
    img = Image.open(io.BytesIO(img_data)).convert('L')  # Convert to grayscale

    # Resize image to 28x28 as required by the MNIST model
    img = img.resize((28, 28))

    # Convert image to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255

    # Expand dimensions to match the input shape of the model (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

    # Predict the digit using the model
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    return jsonify({'digit': int(predicted_digit)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
