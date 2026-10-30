
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("naive_bayes_model.pkl")

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form.get("message", "").strip()

    if not message:
        return jsonify({
            "success": False,
            "message": "Please enter a message."
        })

    # Convert message into features
    message_vector = vectorizer.transform([message])

    # Predict
    prediction = model.predict(message_vector)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(message_vector)[0]

    # Class names
    classes = model.classes_

    probability_data = {}

    for class_name, probability in zip(classes, probabilities):
        probability_data[class_name] = round(
            float(probability) * 100, 2
        )

    confidence = round(
        float(max(probabilities)) * 100,
        2
    )

    if prediction == "Important":

        result = "IMPORTANT"

        result_message = (
            "This message contains important college-related information."
        )

    else:

        result = "PROMOTIONAL"

        result_message = (
            "This message appears to be promotional or an offer."
        )

    return jsonify({

        "success": True,

        "message": message,

        "prediction": prediction,

        "result": result,

        "result_message": result_message,

        "confidence": confidence,

        "probabilities": probability_data

    })


if __name__ == "__main__":
    app.run(debug=True)