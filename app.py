from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os
from glob import glob

app = Flask(__name__)

# Load the trained model
model = load_model("C:/Users/purna/OneDrive/Desktop/project/Lenet/model.h5")

# Function for image prediction
def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    class_indices = {
        0: 'Algaonemo',
        1: 'Alovera',
        2: 'Banyan',
        3: 'Coordyline fruticosa',
        4: 'Custard apple',
        5: 'Eucalyptus',
        6: 'Ficus',
        7: 'Guava',
        8: 'sacred fig',
        9: 'wee ping fig'
    }

    prediction_class = class_indices[np.argmax(result)]
    return prediction_class

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        file_path = os.path.join('C:/Users/purna/OneDrive/Desktop/project/Test/', filename)
        file.save(file_path)

        prediction = predict_image(file_path)

        return jsonify({'filename': filename, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
