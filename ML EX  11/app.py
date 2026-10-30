
import os
import pickle
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

app = Flask(__name__)

# Create folders automatically
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Saved model files
MODEL_FILES = {
    "dt_classifier": "models/decision_tree_classifier.pkl",
    "rf_classifier": "models/random_forest_classifier.pkl",
    "dt_regressor": "models/decision_tree_regressor.pkl",
    "rf_regressor": "models/random_forest_regressor.pkl"
}

# The website currently accepts three input features.
N_FEATURES = 3


def load_dataset(filename, task):
    """
    Load a CSV from the data folder.
    If it does not exist, use a built-in scikit-learn dataset.
    The last CSV column is treated as the target.
    """

    filepath = os.path.join("data", filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)

        if df.shape[1] < 4:
            raise ValueError(
                f"{filename} must contain at least 3 numeric features "
                "and 1 target column."
            )

        # By default, the last column is the target.
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Keep the first three numeric input features.
        X = X.select_dtypes(include=[np.number])
        X = X.iloc[:, :N_FEATURES]

        if X.shape[1] != N_FEATURES:
            raise ValueError(
                f"{filename} must contain at least 3 numeric features."
            )

        if task == "regression":
            y = pd.to_numeric(y, errors="coerce")
            valid = y.notna()
            X = X.loc[valid]
            y = y.loc[valid]

        else:
            valid = y.notna()
            X = X.loc[valid]
            y = y.loc[valid]

        if len(X) < 10 or y.nunique() < 2:
            raise ValueError(
                f"{filename} does not contain enough valid data "
                "or target classes."
            )

        return X, y, list(X.columns)

    # Built-in datasets allow the application to run immediately.
    if task == "classification":
        dataset = load_breast_cancer()
    else:
        dataset = load_diabetes()

    X = pd.DataFrame(
        dataset.data[:, :N_FEATURES],
        columns=dataset.feature_names[:N_FEATURES]
    )
    y = pd.Series(dataset.target, name="target")

    print(f"Using built-in {task} dataset: {dataset.DESCR.splitlines()[0]}")
    return X, y, list(X.columns)


def create_pipeline(model):
    # Imputation is fitted only on the training data.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])


def train_models():
    print("\nTraining machine learning models...")

    # ---------------- CLASSIFICATION ----------------
    Xc, yc, class_features = load_dataset(
        "classification.csv", "classification"
    )

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc,
        test_size=0.20,
        random_state=42,
        stratify=yc
    )

    dtc = create_pipeline(
        DecisionTreeClassifier(random_state=42)
    )
    rfc = create_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )

    dtc.fit(Xc_train, yc_train)
    rfc.fit(Xc_train, yc_train)

    dtc_pred = dtc.predict(Xc_test)
    rfc_pred = rfc.predict(Xc_test)

    classification_metrics = {
        "dt_class_accuracy": round(accuracy_score(yc_test, dtc_pred), 4),
        "rf_class_accuracy": round(accuracy_score(yc_test, rfc_pred), 4),
        "dt_precision": round(precision_score(
            yc_test, dtc_pred, average="weighted", zero_division=0
        ), 4),
        "rf_precision": round(precision_score(
            yc_test, rfc_pred, average="weighted", zero_division=0
        ), 4),
        "dt_recall": round(recall_score(
            yc_test, dtc_pred, average="weighted", zero_division=0
        ), 4),
        "rf_recall": round(recall_score(
            yc_test, rfc_pred, average="weighted", zero_division=0
        ), 4),
        "dt_f1": round(f1_score(
            yc_test, dtc_pred, average="weighted", zero_division=0
        ), 4),
        "rf_f1": round(f1_score(
            yc_test, rfc_pred, average="weighted", zero_division=0
        ), 4)
    }

    # ---------------- REGRESSION ----------------
    Xr, yr, reg_features = load_dataset(
        "regression.csv", "regression"
    )

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr,
        test_size=0.20,
        random_state=42
    )

    dtr = create_pipeline(
        DecisionTreeRegressor(random_state=42)
    )
    rfr = create_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    )

    dtr.fit(Xr_train, yr_train)
    rfr.fit(Xr_train, yr_train)

    dtr_pred = dtr.predict(Xr_test)
    rfr_pred = rfr.predict(Xr_test)

    def rmse(actual, predicted):
        return np.sqrt(mean_squared_error(actual, predicted))

    regression_metrics = {
        "dt_mae": round(mean_absolute_error(yr_test, dtr_pred), 4),
        "rf_mae": round(mean_absolute_error(yr_test, rfr_pred), 4),
        "dt_rmse": round(rmse(yr_test, dtr_pred), 4),
        "rf_rmse": round(rmse(yr_test, rfr_pred), 4),
        "dt_r2": round(r2_score(yr_test, dtr_pred), 4),
        "rf_r2": round(r2_score(yr_test, rfr_pred), 4)
    }

    # Automatically save all four trained models.
    models = {
        "dt_classifier": dtc,
        "rf_classifier": rfc,
        "dt_regressor": dtr,
        "rf_regressor": rfr
    }

    for name, model in models.items():
        with open(MODEL_FILES[name], "wb") as file:
            pickle.dump(model, file)

    print("All four models trained and saved successfully!")

    return (
        models,
        class_features,
        reg_features,
        classification_metrics,
        regression_metrics
    )


# Train and save models when the application starts.
(
    models,
    class_features,
    reg_features,
    classification_metrics,
    regression_metrics
) = train_models()


@app.route("/")
def home():
    metrics = {
        **classification_metrics,
        **regression_metrics
    }

    return render_template(
        "index.html",
        **metrics
    )


@app.route("/predict_classification", methods=["POST"])
def predict_classification():
    try:
        values = [
            float(request.form[f"feature{i}"])
            for i in range(1, N_FEATURES + 1)
        ]

        model_name = request.form.get(
            "model", "random_forest"
        )

        if model_name == "decision_tree":
            model = models["dt_classifier"]
            algorithm = "Decision Tree"
        else:
            model = models["rf_classifier"]
            algorithm = "Random Forest"

        prediction = model.predict([values])[0]

        return render_template(
            "index.html",
            **classification_metrics,
            **regression_metrics,
            classification_result=(
                f"{algorithm} prediction: {prediction}"
            )
        )

    except (ValueError, KeyError) as error:
        return f"Invalid input: {error}", 400


@app.route("/predict_regression", methods=["POST"])
def predict_regression():
    try:
        values = [
            float(request.form[f"feature{i}"])
            for i in range(1, N_FEATURES + 1)
        ]

        model_name = request.form.get(
            "model", "random_forest"
        )

        if model_name == "decision_tree":
            model = models["dt_regressor"]
            algorithm = "Decision Tree"
        else:
            model = models["rf_regressor"]
            algorithm = "Random Forest"

        prediction = float(model.predict([values])[0])

        return render_template(
            "index.html",
            **classification_metrics,
            **regression_metrics,
            regression_result=(
                f"{algorithm} predicted value: {prediction:.4f}"
            )
        )

    except (ValueError, KeyError) as error:
        return f"Invalid input: {error}", 400


if __name__ == "__main__":
    app.run(debug=True)

