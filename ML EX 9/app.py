
from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

MODEL_FILE = "svm_model.pkl"

# UCI Banknote Authentication dataset
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00267/"
    "data_banknote_authentication.txt"
)

FEATURES = [
    "variance",
    "skewness",
    "curtosis",
    "entropy"
]


def train_model():
    # Download dataset
    columns = FEATURES + ["class"]

    data = pd.read_csv(
        DATA_URL,
        header=None,
        names=columns
    )

    X = data[FEATURES]
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale features and train Linear SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear"))
    ])

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    # Save trained model
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)

    print(f"Model trained successfully!")
    print(f"Test accuracy: {accuracy * 100:.2f}%")


# Train only if the saved model does not exist
if not os.path.exists(MODEL_FILE):
    train_model()

# Load saved model
with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        variance = float(request.form["variance"])
        skewness = float(request.form["skewness"])
        curtosis = float(request.form["curtosis"])
        entropy = float(request.form["entropy"])

        values = [[
            variance,
            skewness,
            curtosis,
            entropy
        ]]

        prediction = int(model.predict(values)[0])

        if prediction == 0:
            result = "Genuine Banknote"
            status = "genuine"
        else:
            result = "Counterfeit Banknote"
            status = "counterfeit"

        return render_template(
            "index.html",
            result=result,
            status=status,
            variance=variance,
            skewness=skewness,
            curtosis=curtosis,
            entropy=entropy
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid numerical values."
        )


if __name__ == "__main__":
    app.run(debug=True)


import pickle

# Save the trained Linear SVM model
with open("svm_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Linear SVM model saved successfully!")
