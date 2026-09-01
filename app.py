from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Decision Tree model (Gini index)
# Make sure DTModel.pkl is in the SAME folder as this app.py
with open('DTModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature order MUST exactly match the order used during training
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate all required fields are present
        missing = [f for f in FEATURE_ORDER if f not in data or data[f] == '']
        if missing:
            return jsonify({'error': f'Missing values for: {", ".join(missing)}'}), 400

        # Build the feature vector in the correct order
        features = [float(data[f]) for f in FEATURE_ORDER]
        input_array = np.array(features).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        result = {
            'prediction': int(prediction),
            'label': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected',
            'confidence': confidence
        }
        return jsonify(result)

    except ValueError:
        return jsonify({'error': 'All fields must be valid numbers.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
