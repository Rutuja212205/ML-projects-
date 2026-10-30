"""Load and prepare built-in datasets for ML Algorithm Battle Arena."""

import pandas as pd

from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)


# Dataset information and supported tasks
DATASETS = {
    "iris": {
        "name": "Iris",
        "description": "Classify iris flowers using their measurements.",
        "tasks": ["classification", "clustering"],
        "loader": load_iris,
    },
    "wine": {
        "name": "Wine",
        "description": "Classify wine samples using chemical measurements.",
        "tasks": ["classification", "clustering"],
        "loader": load_wine,
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "description": "Classify samples using computed cell measurements.",
        "tasks": ["classification", "clustering"],
        "loader": load_breast_cancer,
    },
    "diabetes": {
        "name": "Diabetes",
        "description": "Predict a continuous disease-progression measure.",
        "tasks": ["regression", "clustering"],
        "loader": load_diabetes,
    },
}


def list_datasets(task_type):
    """Return datasets that support the selected task."""

    task_type = task_type.strip().lower()

    valid_tasks = {"classification", "regression", "clustering"}

    if task_type not in valid_tasks:
        raise ValueError(
            f"Unsupported task '{task_type}'. "
            f"Choose one of: {', '.join(sorted(valid_tasks))}."
        )

    return [
        {
            "key": key,
            "name": info["name"],
            "description": info["description"],
        }
        for key, info in DATASETS.items()
        if task_type in info["tasks"]
    ]


def load_dataset(dataset_key, task_type):
    """Load a dataset and return its features and applicable target."""

    dataset_key = dataset_key.strip().lower()
    task_type = task_type.strip().lower()

    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    info = DATASETS[dataset_key]

    if task_type not in info["tasks"]:
        raise ValueError(
            f"{info['name']} does not support the "
            f"'{task_type}' task."
        )


    raw_data = info["loader"]()

    features = pd.DataFrame(
        raw_data.data,
        columns=raw_data.feature_names,
    )

    # Clustering is unsupervised, so the model must not receive labels.
    target = None

    if task_type in {"classification", "regression"}:
        target = pd.Series(raw_data.target, name="target")

    return {
        "key": dataset_key,
        "name": info["name"],
        "description": info["description"],
        "task_type": task_type,
        "features": features,
        "target": target,
        "target_names": getattr(raw_data, "target_names", None),
    }