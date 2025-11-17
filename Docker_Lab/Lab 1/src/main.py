from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('my_model.keras')
class_labels = ['Class_0', 'Class_1', 'Class_2']  # Wine dataset classes

@app.route('/')
def home():
    return "Welcome to the Wine Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            # Retrieve 13 features from form inputs
            input_features = [float(data[f'feature{i}']) for i in range(13)]
            input_data = np.array(input_features)[np.newaxis, ]
            prediction = model.predict(input_data)
            predicted_class = class_labels[np.argmax(prediction)]
            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
