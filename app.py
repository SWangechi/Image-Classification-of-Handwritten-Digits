from flask import Flask, request, jsonify, render_template
import joblib
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your models
logistic_model = joblib.load('logistic_model.pkl')
knn_model = joblib.load('knn_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# For deep learning models
mlp_model = MLPModel()  # Instantiate your model class
mlp_model.load_state_dict(torch.load('mlp_model.pth'))
mlp_model.eval()

cnn_model = CNNModel()  # Instantiate your CNN class
cnn_model.load_state_dict(torch.load('cnn_model.pth'))
cnn_model.eval()

# Image transformation for inference
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert image to grayscale
    transforms.Resize((28, 28)),  # Resize image to 28x28
    transforms.ToTensor(),  # Convert image to tensor
])

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')  # A simple HTML form to upload images

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)  # Preprocess the image

    # For traditional models, convert image to 1D array if necessary
    image_np = image.numpy().reshape(1, -1)  # Flatten the image for traditional models
    
    # Example predictions
    logistic_pred = logistic_model.predict(image_np)
    knn_pred = knn_model.predict(image_np)
    svm_pred = svm_model.predict(image_np)

    # For deep learning models
    with torch.no_grad():
        mlp_pred = torch.argmax(mlp_model(image).cpu(), dim=1).item()
        cnn_pred = torch.argmax(cnn_model(image).cpu(), dim=1).item()

    return jsonify({
        'logistic_prediction': int(logistic_pred[0]),
        'knn_prediction': int(knn_pred[0]),
        'svm_prediction': int(svm_pred[0]),
        'mlp_prediction': mlp_pred,
        'cnn_prediction': cnn_pred,
    })

if __name__ == '__main__':
    app.run(debug=True)
