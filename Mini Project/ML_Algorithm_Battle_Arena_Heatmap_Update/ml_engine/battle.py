"""Train, evaluate, and compare machine learning algorithms."""

import time

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_engine.algorithms import create_algorithm
from ml_engine.datasets import load_dataset


SCALED_CLASSIFIERS = {"logistic_regression", "knn", "svm"}

RANKING_METRICS = {
    "classification": "accuracy",
    "regression": "r2",
    "clustering": "silhouette_score",
}


def run_battle(
    dataset_key,
    task_type,
    algorithm_keys,
    test_size=0.2,
    random_state=42,
    algorithm_params=None,
):
    """Train and compare selected algorithms."""

    if not isinstance(task_type, str):
        raise ValueError("Task type must be a string.")

    task_type = task_type.strip().lower()

    if task_type not in RANKING_METRICS:
        raise ValueError(f"Unsupported task type: {task_type}")

    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not np.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ValueError("Test size must be between 0 and 1.")

    if isinstance(algorithm_keys, str):
        algorithm_keys = [algorithm_keys]

    if not isinstance(algorithm_keys, list) or not algorithm_keys:
        raise ValueError("Select at least one algorithm.")

    if not all(isinstance(key, str) and key.strip() for key in algorithm_keys):
        raise ValueError("Algorithm keys must be non-empty strings.")

    algorithm_keys = [key.strip().lower() for key in algorithm_keys]

    if len(algorithm_keys) != len(set(algorithm_keys)):
        raise ValueError("Duplicate algorithm selections are not allowed.")

    if algorithm_params is None:
        algorithm_params = {}

    if not isinstance(algorithm_params, dict):
        raise ValueError("Algorithm parameters must be an object.")

    unknown_algorithm_parameters = set(algorithm_params) - set(algorithm_keys)
    if unknown_algorithm_parameters:
        raise ValueError(
            "Parameters were provided for algorithms that were not selected."
        )

    data = load_dataset(dataset_key, task_type)
    features = data["features"]
    target = data["target"]

    configurations = {}
    for key in algorithm_keys:
        model, metadata = create_algorithm(
            task_type, key, algorithm_params.get(key, {})
        )
        configurations[key] = (model, metadata)

    if task_type in {"classification", "regression"}:
        stratify = target if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    results = []

    if task_type == "classification":
        # Keep a stable order for both matrix rows/columns and the displayed labels.
        class_labels = np.unique(target).tolist()

        for key in algorithm_keys:
            model, metadata = configurations[key]

            if key in SCALED_CLASSIFIERS:
                model = make_pipeline(StandardScaler(), model)

            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            training_seconds = time.perf_counter() - start_time
            predictions = model.predict(X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                )),
                "recall": float(recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                )),
                "f1_score": float(f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                )),
                "confusion_matrix": confusion_matrix(
                    y_test, predictions, labels=class_labels
                ).tolist(),
                "class_labels": class_labels,
            }

            results.append({
                "algorithm_key": key,
                "name": metadata["name"],
                "parameters": metadata["configured_parameters"],
                "metrics": metrics,
                "training_seconds": round(training_seconds, 6),
            })

    elif task_type == "regression":
        for key in algorithm_keys:
            model, metadata = configurations[key]

            if metadata.get("feature_mode") == "single":
                X_train_model = X_train.iloc[:, :1]
                X_test_model = X_test.iloc[:, :1]
            else:
                X_train_model = X_train
                X_test_model = X_test

            start_time = time.perf_counter()
            model.fit(X_train_model, y_train)
            training_seconds = time.perf_counter() - start_time
            predictions = model.predict(X_test_model)

            metrics = {
                "r2": float(r2_score(y_test, predictions)),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            }

            results.append({
                "algorithm_key": key,
                "name": metadata["name"],
                "parameters": metadata["configured_parameters"],
                "features_used": list(X_train_model.columns),
                "metrics": metrics,
                "training_seconds": round(training_seconds, 6),
            })

    elif task_type == "clustering":
        scaled_features = StandardScaler().fit_transform(features)

        for key in algorithm_keys:
            model, metadata = configurations[key]
            start_time = time.perf_counter()
            cluster_labels = model.fit_predict(scaled_features)
            training_seconds = time.perf_counter() - start_time

            unique_clusters = np.unique(cluster_labels)
            number_of_clusters = len(unique_clusters)
            metrics = {
                "number_of_clusters": int(number_of_clusters),
                "silhouette_score": None,
                "calinski_harabasz_score": None,
                "davies_bouldin_score": None,
            }

            if 1 < number_of_clusters < len(scaled_features):
                metrics["silhouette_score"] = float(
                    silhouette_score(scaled_features, cluster_labels)
                )
                metrics["calinski_harabasz_score"] = float(
                    calinski_harabasz_score(scaled_features, cluster_labels)
                )
                metrics["davies_bouldin_score"] = float(
                    davies_bouldin_score(scaled_features, cluster_labels)
                )

            results.append({
                "algorithm_key": key,
                "name": metadata["name"],
                "parameters": metadata["configured_parameters"],
                "metrics": metrics,
                "training_seconds": round(training_seconds, 6),
            })

    ranking_metric = RANKING_METRICS[task_type]
    eligible_results = [
        result for result in results
        if result["metrics"].get(ranking_metric) is not None
        and np.isfinite(result["metrics"][ranking_metric])
    ]

    winner = None
    if eligible_results:
        best_result = max(
            eligible_results,
            key=lambda result: result["metrics"][ranking_metric],
        )
        winner = {
            "algorithm_key": best_result["algorithm_key"],
            "name": best_result["name"],
            "metric": ranking_metric,
            "score": best_result["metrics"][ranking_metric],
        }

    return {
        "dataset": data["name"],
        "task_type": task_type,
        "sample_count": len(features),
        "feature_count": features.shape[1],
        "ranking_metric": ranking_metric,
        "winner": winner,
        "results": results,
    }
