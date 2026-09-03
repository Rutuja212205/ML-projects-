from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Multiple Linear Regression model
model = pickle.load(open("MLRModel.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read inputs from the HTML form
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    # Predict package
    prediction = model.predict(np.array([[cgpa, iq]]))

    return render_template(
        "index.html",
        prediction_text=f"Predicted Package : {prediction[0]:.2f} LPA"
    )


if __name__ == "__main__":
    app.run(debug=True)