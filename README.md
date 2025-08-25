# MNIST Digit Recognizer

An interactive web application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can draw digits on a canvas and get real-time predictions.

## 🎯 Features

- **Interactive Drawing Canvas**: Draw digits using an intuitive web interface
- **Real-time Recognition**: Get instant predictions as you draw
- **High Accuracy**: CNN model achieves 98.63% accuracy on test data
- **Customizable Interface**: Adjust stroke width, colors, and drawing tools
- **Confidence Threshold**: Only shows predictions above a certain confidence level

## 🏗️ Architecture

The project consists of two main components:

1. **Model Training** (`MNIST_Train_Model.ipynb`): Jupyter notebook for training the CNN
2. **Web Application** (`app.py`): Streamlit app for interactive digit recognition

### Model Architecture

- **Input Layer**: 28x28 grayscale images
- **Conv2D Layer 1**: 32 filters (3x3) + ReLU + MaxPooling
- **Conv2D Layer 2**: 64 filters (3x3) + ReLU + MaxPooling  
- **Conv2D Layer 3**: 64 filters (3x3) + ReLU
- **Dense Layer**: 64 neurons + ReLU
- **Output Layer**: 10 neurons (softmax) for digit classification

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mnist-digit-recognizer.git
   cd mnist-digit-recognizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (Optional - if model file not provided)
   ```bash
   jupyter notebook MNIST_Train_Model.ipynb
   ```
   Run all cells to train and save the model as `my_mnist_model.h5`

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📋 Requirements

```
streamlit-drawable-canvas
numpy
tensorflow
Pillow
flask
```

## 🎮 Usage

1. **Launch the application** using the command above
2. **Draw a digit** (0-9) on the black canvas using your mouse
3. **Adjust settings** in the sidebar:
   - Stroke width (1-25 pixels)
   - Stroke color
   - Background color
   - Drawing tools (freedraw, line, rect, circle, transform)
4. **View prediction** - The model will display the predicted digit and confidence score
5. **Clear and redraw** to try different digits

## 🔧 Configuration

### Confidence Threshold
The default confidence threshold is set to 0.5 (50%). You can modify this in `app.py`:

```python
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed
```

### Canvas Settings
Default canvas size is 280x280 pixels. Drawings are automatically resized to 28x28 for model input.

## 📊 Model Performance

- **Training Accuracy**: ~99.33%
- **Test Accuracy**: 98.63%
- **Model Size**: 364.55 KB
- **Total Parameters**: 93,322

## 🗂️ Project Structure

```
mnist-digit-recognizer/
│
├── app.py                      # Streamlit web application
├── MNIST_Train_Model.ipynb     # Model training notebook
├── requirements.txt            # Python dependencies
├── my_mnist_model.h5          # Trained model file
└── README.md                  # Project documentation
```

## 🛠️ Technical Details

### Data Processing Pipeline
1. Canvas input (280x280 RGBA) → Grayscale conversion
2. Resize to 28x28 pixels (MNIST standard)
3. Normalize pixel values (0-1 range)
4. Reshape for model input (1, 28, 28, 1)

### Model Training Details
- **Dataset**: MNIST (60,000 training + 10,000 test images)
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Epochs**: 5
- **Batch Size**: 64

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Database**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **TensorFlow/Keras**: For the deep learning framework
- **Streamlit**: For the web application framework
- **streamlit-drawable-canvas**: For the interactive drawing component

## 📞 Contact

Your Name - jalalkhanscience@gmail.com

Project Link: [https://github.com/Jalalkhan912/mnist-digit-recognizer](https://github.com/Jalalkhan912/MNIST-Deployement)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
