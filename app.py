from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
from model import load_model, predict

app = Flask(__name__)

# Charger le modèle de deep learning
model = load_model()

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Décoder l'image en base64
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Prétraitement pour le modèle
        input_data = np.array(image.resize((224, 224))) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # Faire une prédiction
        prediction = predict(model, input_data)
        result = 'Real' if prediction[0] > 0.5 else 'Spoof'

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
