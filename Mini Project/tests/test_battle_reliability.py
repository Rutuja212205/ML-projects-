"""Reliability tests for ML Algorithm Battle Arena.

These tests check that:
1. Classification, regression, and clustering battles can run.
2. Results contain valid numerical metrics.
3. Repeating an experiment with the same configuration produces
   the same metric values.
"""

import json
import math
import unittest

from ml_engine.algorithms import list_algorithms
from ml_engine.battle import run_battle
from ml_engine.datasets import list_datasets


def normalize_for_comparison(value):
    """Convert results into JSON-compatible values for comparison."""
    if hasattr(value, "tolist"):
        return normalize_for_comparison(value.tolist())

    if isinstance(value, dict):
        return {
            str(key): normalize_for_comparison(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize_for_comparison(item) for item in value]

    if hasattr(value, "item"):
        try:
            return normalize_for_comparison(value.item())
        except (ValueError, TypeError):
            pass

    return value


def numeric_values(value):
    """Collect finite numeric values from nested result data."""
    values = []

    if isinstance(value, dict):
        for item in value.values():
            values.extend(numeric_values(item))

    elif isinstance(value, (list, tuple)):
        for item in value:
            values.extend(numeric_values(item))

    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        values.append(float(value))

    else:
        try:
            if hasattr(value, "item"):
                converted = value.item()

                if isinstance(converted, (int, float)):
                    values.append(float(converted))
        except (ValueError, TypeError):
            pass

    return values


class BattleReliabilityTests(unittest.TestCase):
    """Check deterministic behavior across supported ML task types."""

    TASK_TYPES = (
        "classification",
        "regression",
        "clustering",
    )

    def run_and_check_battle(self, task_type):
        """Run the same battle twice and compare its metrics."""

        available_datasets = list_datasets(task_type)
        available_algorithms = list_algorithms(task_type)

        self.assertTrue(
            available_datasets,
            f"No datasets were found for {task_type}.",
        )

        self.assertTrue(
            available_algorithms,
            f"No algorithms were found for {task_type}.",
        )

        dataset_key = available_datasets[0]["key"]

        # Test two available algorithms where possible.
        algorithm_keys = [
            algorithm["key"]
            for algorithm in available_algorithms[:2]
        ]

        configuration = {
            "dataset_key": dataset_key,
            "task_type": task_type,
            "algorithm_keys": algorithm_keys,
            "test_size": 0.2,
            "random_state": 42,
        }

        first_run = run_battle(
            dataset_key=configuration["dataset_key"],
            task_type=configuration["task_type"],
            algorithm_keys=configuration["algorithm_keys"],
            test_size=configuration["test_size"],
            random_state=configuration["random_state"],
        )

        second_run = run_battle(
            dataset_key=configuration["dataset_key"],
            task_type=configuration["task_type"],
            algorithm_keys=configuration["algorithm_keys"],
            test_size=configuration["test_size"],
            random_state=configuration["random_state"],
        )

        # Both experiments should return algorithm results.
        self.assertIn("results", first_run)
        self.assertIn("results", second_run)

        first_results = first_run["results"]
        second_results = second_run["results"]

        self.assertTrue(
            first_results,
            f"{task_type}: the battle returned no results.",
        )

        self.assertEqual(
            len(first_results),
            len(second_results),
            f"{task_type}: repeated runs returned different "
            "numbers of algorithm results.",
        )

        # Each algorithm should return a non-empty metrics dictionary.
        for result in first_results:
            self.assertIn("metrics", result)

            metrics = result["metrics"]

            self.assertIsInstance(metrics, dict)

            self.assertTrue(
                metrics,
                f"{task_type}: {result.get('name', 'Algorithm')} "
                "returned no evaluation metrics.",
            )

            values = numeric_values(metrics)

            self.assertTrue(
                values,
                f"{task_type}: no numerical metrics were returned "
                f"by {result.get('name', 'an algorithm')}.",
            )

            for value in values:
                self.assertTrue(
                    math.isfinite(value),
                    f"{task_type}: a metric is not finite: {value}",
                )

        # Compare metrics only. Training time can naturally vary
        # between runs, so it should not affect this comparison.
        first_metrics = [
            {
                "name": result.get("name"),
                "metrics": normalize_for_comparison(result["metrics"]),
            }
            for result in first_results
        ]

        second_metrics = [
            {
                "name": result.get("name"),
                "metrics": normalize_for_comparison(result["metrics"]),
            }
            for result in second_results
        ]

        first_json = json.dumps(
            first_metrics,
            sort_keys=True,
            allow_nan=False,
        )

        second_json = json.dumps(
            second_metrics,
            sort_keys=True,
            allow_nan=False,
        )

        self.assertEqual(
            first_json,
            second_json,
            f"{task_type}: repeated runs with the same random seed "
            "produced different metrics.",
        )

        return dataset_key, len(first_results)

    def test_classification_is_repeatable(self):
        dataset, count = self.run_and_check_battle("classification")

        print(
            f"\nPASS: Classification | "
            f"Dataset: {dataset} | Algorithms: {count}"
        )

    def test_regression_is_repeatable(self):
        dataset, count = self.run_and_check_battle("regression")

        print(
            f"\nPASS: Regression | "
            f"Dataset: {dataset} | Algorithms: {count}"
        )

    def test_clustering_is_repeatable(self):
        dataset, count = self.run_and_check_battle("clustering")

        print(
            f"\nPASS: Clustering | "
            f"Dataset: {dataset} | Algorithms: {count}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)