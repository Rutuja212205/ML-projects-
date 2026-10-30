"""Flask application for ML Algorithm Battle Arena."""

from flask import Flask, jsonify, render_template, request

from ml_engine.algorithms import list_algorithms
from ml_engine.battle import run_battle
from ml_engine.datasets import list_datasets
from ml_engine.history import (
    get_recent_battles,
    initialize_history_db,
    save_battle,
)


app = Flask(__name__)

initialize_history_db()


# ======================================================
# HOME PAGE
# ======================================================

@app.route("/")
def home():
    return render_template("index.html")


# ======================================================
# DATASET API
# ======================================================

@app.route("/api/datasets", methods=["GET"])
def get_datasets():

    task_type = request.args.get(
        "task_type",
        "classification",
    )

    try:
        return jsonify({
            "datasets": list_datasets(task_type)
        })

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


# ======================================================
# ALGORITHM API
# ======================================================

@app.route("/api/algorithms", methods=["GET"])
def get_algorithms():

    task_type = request.args.get(
        "task_type",
        "classification",
    )

    try:
        return jsonify({
            "algorithms": list_algorithms(task_type)
        })

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


# ======================================================
# BATTLE API
# ======================================================

@app.route("/api/battle", methods=["POST"])
def start_battle():

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "A valid JSON request is required."
        }), 400

    dataset_key = payload.get("dataset")
    task_type = payload.get("task_type")
    algorithm_keys = payload.get("algorithms")
    algorithm_params = payload.get("algorithm_params", {})

    if not isinstance(dataset_key, str) or not dataset_key.strip():
        return jsonify({
            "error": "Select a valid dataset."
        }), 400

    if not isinstance(task_type, str) or not task_type.strip():
        return jsonify({
            "error": "Select a learning task."
        }), 400

    if (
        not isinstance(algorithm_keys, list)
        or not algorithm_keys
    ):
        return jsonify({
            "error": "Select at least one algorithm."
        }), 400

    if not all(
        isinstance(key, str) and key.strip()
        for key in algorithm_keys
    ):
        return jsonify({
            "error": "Algorithm selections must be valid names."
        }), 400

    if not isinstance(algorithm_params, dict):
        return jsonify({
            "error": "Algorithm parameters must be an object."
        }), 400

    try:

        test_size = float(
            payload.get("test_size", 0.2)
        )

        result = run_battle(
            dataset_key=dataset_key,
            task_type=task_type,
            algorithm_keys=algorithm_keys,
            test_size=test_size,
            algorithm_params=algorithm_params,
        )

        battle_id = save_battle(result)

        result["battle_id"] = battle_id

        return jsonify(result)

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception(
            "Unexpected error while running or saving a battle."
        )

        return jsonify({
            "error": (
                "The battle could not be completed or its "
                "results could not be saved."
            )
        }), 500


# ======================================================
# BATTLE HISTORY API
# ======================================================

@app.route("/api/history", methods=["GET"])
def battle_history():

    try:
        limit = int(
            request.args.get("limit", "10")
        )

        if not 1 <= limit <= 100:
            return jsonify({
                "error": "Limit must be between 1 and 100."
            }), 400

        battles = get_recent_battles(limit)

        return jsonify({
            "count": len(battles),
            "battles": battles,
        })

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception(
            "Could not retrieve battle history."
        )

        return jsonify({
            "error": "Could not retrieve battle history."
        }), 500


# ======================================================
# RUN APPLICATION
# ======================================================

if __name__ == "__main__":
    app.run(debug=True)