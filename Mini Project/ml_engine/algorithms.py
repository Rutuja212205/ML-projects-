"""Algorithm registry with configurable parameters."""

import math

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_engine.bayesian_network import BayesianNetworkClassifier


ALGORITHM_REGISTRY = {
    "classification": {
        "logistic_regression": {
            "name": "Logistic Regression",
            "estimator": LogisticRegression,
            "params": {
                "max_iter": 2000,
                "C": 1.0,
            },
            "parameter_schema": [
                {
                    "key": "max_iter",
                    "label": "Maximum iterations",
                    "type": "integer",
                    "default": 2000,
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                },
                {
                    "key": "C",
                    "label": "Regularization strength",
                    "type": "number",
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1,
                },
            ],
        },

        "knn": {
            "name": "K-Nearest Neighbors",
            "estimator": KNeighborsClassifier,
            "params": {
                "n_neighbors": 5,
                "weights": "uniform",
            },
            "parameter_schema": [
                {
                    "key": "n_neighbors",
                    "label": "Number of neighbors",
                    "type": "integer",
                    "default": 5,
                    "min": 1,
                    "max": 25,
                    "step": 1,
                },
                {
                    "key": "weights",
                    "label": "Neighbor weights",
                    "type": "choice",
                    "default": "uniform",
                    "options": [
                        {"label": "Uniform", "value": "uniform"},
                        {"label": "Distance", "value": "distance"},
                    ],
                },
            ],
        },

        "naive_bayes": {
            "name": "Naive Bayes",
            "estimator": GaussianNB,
            "params": {
                "var_smoothing": 1e-9,
            },
            "parameter_schema": [
                {
                    "key": "var_smoothing",
                    "label": "Variance smoothing",
                    "type": "number",
                    "default": 1e-9,
                    "min": 1e-12,
                    "max": 1e-3,
                    "step": "any",
                },
            ],
        },

        "bayesian_network": {
            "name": "Bayesian Network",
            "estimator": BayesianNetworkClassifier,
            "params": {
                "n_bins": 4,
                "alpha": 1.0,
            },
            "parameter_schema": [
                {
                    "key": "n_bins",
                    "label": "Number of discretization bins",
                    "type": "integer",
                    "default": 4,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                },
                {
                    "key": "alpha",
                    "label": "Laplace smoothing",
                    "type": "number",
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                },
            ],
        },

        "decision_tree": {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier,
            "params": {
                "random_state": 42,
                "max_depth": None,
                "criterion": "gini",
                "min_samples_split": 2,
            },
            "parameter_schema": [
                {
                    "key": "max_depth",
                    "label": "Maximum tree depth",
                    "type": "choice",
                    "default": None,
                    "options": [
                        {"label": "Unlimited", "value": None},
                        {"label": "2", "value": 2},
                        {"label": "3", "value": 3},
                        {"label": "5", "value": 5},
                        {"label": "8", "value": 8},
                        {"label": "12", "value": 12},
                    ],
                },
                {
                    "key": "criterion",
                    "label": "Split criterion",
                    "type": "choice",
                    "default": "gini",
                    "options": [
                        {"label": "Gini", "value": "gini"},
                        {"label": "Entropy", "value": "entropy"},
                    ],
                },
                {
                    "key": "min_samples_split",
                    "label": "Minimum samples to split",
                    "type": "integer",
                    "default": 2,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                },
            ],
        },

        "svm": {
            "name": "Support Vector Machine",
            "estimator": SVC,
            "params": {
                "random_state": 42,
                "C": 1.0,
                "kernel": "rbf",
            },
            "parameter_schema": [
                {
                    "key": "C",
                    "label": "Regularization strength",
                    "type": "number",
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.1,
                },
                {
                    "key": "kernel",
                    "label": "Kernel",
                    "type": "choice",
                    "default": "rbf",
                    "options": [
                        {"label": "Linear", "value": "linear"},
                        {"label": "RBF", "value": "rbf"},
                        {"label": "Polynomial", "value": "poly"},
                        {"label": "Sigmoid", "value": "sigmoid"},
                    ],
                },
            ],
        },
    },

    "regression": {
        "simple_linear_regression": {
            "name": "Simple Linear Regression",
            "estimator": LinearRegression,
            "params": {
                "fit_intercept": True,
            },
            "feature_mode": "single",
            "parameter_schema": [
                {
                    "key": "fit_intercept",
                    "label": "Fit intercept",
                    "type": "boolean",
                    "default": True,
                },
            ],
        },

        "multivariate_linear_regression": {
            "name": "Multivariate Linear Regression",
            "estimator": LinearRegression,
            "params": {
                "fit_intercept": True,
            },
            "feature_mode": "all",
            "parameter_schema": [
                {
                    "key": "fit_intercept",
                    "label": "Fit intercept",
                    "type": "boolean",
                    "default": True,
                },
            ],
        },
    },

    "clustering": {
        "kmeans": {
            "name": "K-Means Clustering",
            "estimator": KMeans,
            "params": {
                "n_clusters": 3,
                "random_state": 42,
                "n_init": 10,
            },
            "parameter_schema": [
                {
                    "key": "n_clusters",
                    "label": "Number of clusters",
                    "type": "integer",
                    "default": 3,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                },
                {
                    "key": "n_init",
                    "label": "Initialization runs",
                    "type": "integer",
                    "default": 10,
                    "min": 5,
                    "max": 30,
                    "step": 1,
                },
            ],
        },

        "agglomerative": {
            "name": "Agglomerative Clustering",
            "estimator": AgglomerativeClustering,
            "params": {
                "n_clusters": 3,
                "linkage": "ward",
            },
            "parameter_schema": [
                {
                    "key": "n_clusters",
                    "label": "Number of clusters",
                    "type": "integer",
                    "default": 3,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                },
                {
                    "key": "linkage",
                    "label": "Linkage method",
                    "type": "choice",
                    "default": "ward",
                    "options": [
                        {"label": "Ward", "value": "ward"},
                        {"label": "Complete", "value": "complete"},
                        {"label": "Average", "value": "average"},
                        {"label": "Single", "value": "single"},
                    ],
                },
            ],
        },
    },
}


def list_algorithms(task_type):
    """Return available algorithms and their configurable parameters."""

    if not isinstance(task_type, str):
        raise ValueError("Task type must be a string.")

    task_type = task_type.strip().lower()

    if task_type not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unsupported task type: {task_type}")

    algorithms = []

    for key, configuration in ALGORITHM_REGISTRY[task_type].items():
        algorithms.append({
            "key": key,
            "name": configuration["name"],
            "parameters": configuration.get("parameter_schema", []),
        })

    return algorithms


def create_algorithm(task_type, algorithm_key, parameter_overrides=None):
    """Create an algorithm using validated parameter overrides."""

    if not isinstance(task_type, str):
        raise ValueError("Task type must be a string.")

    if not isinstance(algorithm_key, str):
        raise ValueError("Algorithm key must be a string.")

    task_type = task_type.strip().lower()
    algorithm_key = algorithm_key.strip().lower()

    if task_type not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unsupported task type: {task_type}")

    algorithms = ALGORITHM_REGISTRY[task_type]

    if algorithm_key not in algorithms:
        raise ValueError(
            f"Algorithm '{algorithm_key}' is not available "
            f"for task '{task_type}'."
        )

    if parameter_overrides is None:
        parameter_overrides = {}

    if not isinstance(parameter_overrides, dict):
        raise ValueError("Algorithm parameters must be an object.")

    configuration = algorithms[algorithm_key]

    schema_list = configuration.get("parameter_schema", [])
    schema_by_key = {
        item["key"]: item for item in schema_list
    }

    unknown_parameters = (
        set(parameter_overrides) - set(schema_by_key)
    )

    if unknown_parameters:
        raise ValueError(
            "Unsupported parameters for "
            f"{configuration['name']}: "
            f"{', '.join(sorted(unknown_parameters))}"
        )

    # Start with the trusted defaults.
    parameters = configuration.get("params", {}).copy()

    # Validate every parameter supplied by the user.
    for key, value in parameter_overrides.items():
        schema = schema_by_key[key]
        parameter_type = schema["type"]

        if parameter_type == "integer":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or int(value) != value
            ):
                raise ValueError(
                    f"{schema['label']} must be a whole number."
                )

            value = int(value)

        elif parameter_type == "number":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{schema['label']} must be a finite number."
                )

            value = float(value)

        elif parameter_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(
                    f"{schema['label']} must be true or false."
                )

        elif parameter_type == "choice":
            allowed_values = [
                option["value"]
                for option in schema["options"]
            ]

            if value not in allowed_values:
                raise ValueError(
                    f"Invalid value for {schema['label']}."
                )

        else:
            raise ValueError(
                f"Unsupported parameter type: {parameter_type}"
            )

        if "min" in schema and value < schema["min"]:
            raise ValueError(
                f"{schema['label']} must be at least {schema['min']}."
            )

        if "max" in schema and value > schema["max"]:
            raise ValueError(
                f"{schema['label']} must be at most {schema['max']}."
            )

        parameters[key] = value

    model = configuration["estimator"](**parameters)

    metadata = {
        key: value
        for key, value in configuration.items()
        if key not in {"estimator", "params"}
    }

    # Preserve the values used for this model in the battle results.
    metadata["configured_parameters"] = parameters.copy()

    return model, metadata