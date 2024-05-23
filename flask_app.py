from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('mnist_model.h5')
print("Model loaded successfully")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'})

    if file:
        print("File received:", file.filename)

        # Read and process the image
        try:
            img = Image.open(io.BytesIO(file.read())).convert('L')
            print("Image opened and converted to grayscale")

            img = img.resize((28, 28))
            print("Image resized to 28x28")

            img_array = np.array(img) / 255.0
            print("Image converted to array and normalized")

            img_array = img_array.reshape(1, 28, 28, 1)
            print("Image reshaped to (1, 28, 28, 1)")

            # Predict using the model
            prediction = model.predict(img_array)
            print("Prediction made:", prediction)

            predicted_number = np.argmax(prediction)
            confidence = np.max(prediction)  # Get the maximum probability
            print("Predicted number:", predicted_number,
                  "Confidence:", confidence)

            # Return the entire prediction array
            return jsonify({'guess': int(predicted_number), 'confidence': prediction[0].tolist()})
        except Exception as e:
            print("Error processing the image:", e)
            return jsonify({'error': 'Error processing the image'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
