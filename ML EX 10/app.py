
from flask import Flask, render_template, request
import os
import pickle

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Model file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RBF_FILE = os.path.join(BASE_DIR, "nonlinear_svm_model.pkl")
POLY_FILE = os.path.join(BASE_DIR, "polynomial_svm_model.pkl")


# Automatically create and save models
def create_models():
    if os.path.exists(RBF_FILE) and os.path.exists(POLY_FILE):
        print("Saved SVM models found. Loading existing models.")
        return

    print("Creating dataset and training SVM models...")

    # Create non-linear dataset
    X, y = make_moons(
        n_samples=500,
        noise=0.20,
        random_state=42
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create models
    models = {
        RBF_FILE: make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1, gamma="scale")
        ),
        POLY_FILE: make_pipeline(
            StandardScaler(),
            SVC(kernel="poly", degree=3, C=1)
        )
    }

    # Train, evaluate, and save models
    for filename, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(
            os.path.basename(filename),
            "Accuracy:",
            f"{accuracy * 100:.2f}%"
        )

        with open(filename, "wb") as file:
            pickle.dump(model, file)

        print(os.path.basename(filename), "created successfully!")

    print("Both SVM models are ready!")


# Load saved models
def load_models():
    with open(RBF_FILE, "rb") as file:
        rbf_model = pickle.load(file)

    with open(POLY_FILE, "rb") as file:
        poly_model = pickle.load(file)

    return rbf_model, poly_model


# Train automatically when model files are missing
create_models()
rbf_model, poly_model = load_models()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature1 = float(request.form["feature1"])
        feature2 = float(request.form["feature2"])

        kernel = request.form.get("kernel", "rbf")

        if kernel == "poly":
            model = poly_model
            model_name = "Polynomial Kernel"
        else:
            model = rbf_model
            model_name = "RBF Kernel"

        prediction = int(
            model.predict([[feature1, feature2]])[0]
        )

        result = f"Class {prediction}"
        status = "class-zero" if prediction == 0 else "class-one"

        return render_template(
            "index.html",
            result=result,
            status=status,
            model_name=model_name,
            feature1=feature1,
            feature2=feature2,
            selected_kernel=kernel
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid numerical feature values."
        )


if __name__ == "__main__":
    app.run(debug=True)


import pickle

# Save the trained RBF SVM model
with open("nonlinear_svm_model.pkl", "wb") as file:
    pickle.dump(rbf_model, file)

# Save the trained Polynomial SVM model
with open("polynomial_svm_model.pkl", "wb") as file:
    pickle.dump(poly_model, file)

print("Both SVM models saved successfully!")