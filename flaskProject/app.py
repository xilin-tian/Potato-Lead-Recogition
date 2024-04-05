from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)

# Load your model (make sure the path is correct and accessible)
model = load_model('/Users/chuyue/Desktop/ori_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in the request.", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file for uploading.", 400

    if file:
        try:
            # Convert the file to an image and preprocess it
            # Read the file into a BytesIO stream
            img_bytes = io.BytesIO(file.read())
            img = image.load_img(img_bytes, target_size=(224, 224))

            img_array = image.img_to_array(img)
            img_array /= 255.0  # Normalize the image data to 0-1 range
            img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            class_labels = ['Healthy', 'Early Blight', 'Late Blight']
            prediction_result = class_labels[predicted_class_index]

            return render_template('result.html', result=prediction_result)
        except Exception as e:
            print(e)
            return "Error processing the image.", 500

    return "There was an error processing the file.", 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)

