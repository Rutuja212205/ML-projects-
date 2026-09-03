from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Trained Model
model = pickle.load(open("MCModel.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read input values
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Prediction
    prediction = model.predict([[sepal_length,
                                 sepal_width,
                                 petal_length,
                                 petal_width]])

    # Convert class number into species name
    if prediction[0] == 0:
        flower = "Iris Setosa"

    elif prediction[0] == 1:
        flower = "Iris Versicolor"

    else:
        flower = "Iris Virginica"

    return render_template(
        "index.html",
        prediction_text=f"Predicted Species : {flower}"
    )


if __name__ == "__main__":
    app.run(debug=True)