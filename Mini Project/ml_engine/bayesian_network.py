"""
Bayesian Network Classifier.

Uses a discrete Bayesian network with:
- A class node connected to every feature.
- A chain of dependencies between consecutive features.
- Quantile discretization for continuous input features.
- Laplace smoothing to handle unseen feature combinations.

This is a fixed network structure, not an automatic
Bayesian network structure-learning algorithm.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class BayesianNetworkClassifier(ClassifierMixin, BaseEstimator):
    """Classify observations using a discrete Bayesian network."""

    def __init__(self, n_bins=4, alpha=1.0):
        self.n_bins = n_bins
        self.alpha = alpha

    @staticmethod
    def _validate_features(X):
        """Validate and convert the feature matrix."""

        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Bayesian Network features must be numeric."
            ) from error

        if X.ndim != 2:
            raise ValueError(
                "Features must be a two-dimensional matrix."
            )

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                "The feature matrix cannot be empty."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Features cannot contain missing or infinite values."
            )

        return X

    def _discretize(self, X):
        """Convert continuous values into discrete feature bins."""

        X_binned = np.empty(X.shape, dtype=np.int64)

        for column_index, edges in enumerate(self.bin_edges_):
            X_binned[:, column_index] = np.searchsorted(
                edges,
                X[:, column_index],
                side="right",
            )

        return X_binned

    def fit(self, X, y):
        """Estimate network probabilities from training observations."""

        X = self._validate_features(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(
                "The target must contain one label per observation."
            )

        if len(X) != len(y):
            raise ValueError(
                "Features and targets must have the same number of rows."
            )

        if not 2 <= self.n_bins <= 20:
            raise ValueError(
                "n_bins must be between 2 and 20."
            )

        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(
                "alpha must be a positive finite number."
            )

        if y.size == 0:
            raise ValueError(
                "The target cannot be empty."
            )

        if np.issubdtype(y.dtype, np.number):
            if not np.isfinite(y).all():
                raise ValueError(
                    "Target labels cannot contain missing or infinite values."
                )
        elif any(
            value is None or (
                isinstance(value, str) and not value.strip()
            )
            for value in y
        ):
            raise ValueError(
                "Target labels cannot be missing or empty."
            )

        self.classes_, y_encoded = np.unique(
            y,
            return_inverse=True,
        )

        if len(self.classes_) < 2:
            raise ValueError(
                "Classification requires at least two different classes."
            )

        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)

        # Create discretization boundaries using training data only.
        quantiles = np.linspace(
            0,
            1,
            self.n_bins + 1,
        )[1:-1]

        self.bin_edges_ = [
            np.unique(
                np.quantile(X[:, column_index], quantiles)
            )
            for column_index in range(self.n_features_in_)
        ]

        self.bin_counts_ = np.asarray(
            [len(edges) + 1 for edges in self.bin_edges_],
            dtype=np.int64,
        )

        X_binned = self._discretize(X)

        # Estimate P(class).
        class_counts = np.bincount(
            y_encoded,
            minlength=self.n_classes_,
        )

        class_prior = (
            class_counts + self.alpha
        ) / (
            len(y_encoded)
            + self.alpha * self.n_classes_
        )

        self.class_log_prior_ = np.log(class_prior)

        self.feature_log_probabilities_ = []

        # First feature:
        # P(X0 | class)
        first_feature_bins = self.bin_counts_[0]

        first_counts = np.zeros(
            (self.n_classes_, first_feature_bins),
            dtype=np.float64,
        )

        np.add.at(
            first_counts,
            (y_encoded, X_binned[:, 0]),
            1,
        )

        first_probabilities = (
            first_counts + self.alpha
        ) / (
            first_counts.sum(axis=1, keepdims=True)
            + self.alpha * first_feature_bins
        )

        self.feature_log_probabilities_.append(
            np.log(first_probabilities)
        )

        # Remaining features:
        # P(Xi | class, X(i-1))
        #
        # This introduces dependencies between neighboring
        # features rather than assuming complete independence.

        for column_index in range(1, self.n_features_in_):

            previous_bins = int(
                self.bin_counts_[column_index - 1]
            )

            current_bins = int(
                self.bin_counts_[column_index]
            )

            counts = np.zeros(
                (
                    self.n_classes_,
                    previous_bins,
                    current_bins,
                ),
                dtype=np.float64,
            )

            np.add.at(
                counts,
                (
                    y_encoded,
                    X_binned[:, column_index - 1],
                    X_binned[:, column_index],
                ),
                1,
            )

            probabilities = (
                counts + self.alpha
            ) / (
                counts.sum(axis=2, keepdims=True)
                + self.alpha * current_bins
            )

            self.feature_log_probabilities_.append(
                np.log(probabilities)
            )

        return self

    def _joint_log_scores(self, X):
        """Calculate an unnormalized log score for every class."""

        X = self._validate_features(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"received {X.shape[1]}."
            )

        X_binned = self._discretize(X)

        scores = np.tile(
            self.class_log_prior_,
            (X.shape[0], 1),
        )

        # Add log P(X0 | class).
        scores += self.feature_log_probabilities_[0][
            :,
            X_binned[:, 0],
        ].T

        # Add log P(Xi | class, X(i-1)).
        for column_index in range(1, self.n_features_in_):

            probabilities = self.feature_log_probabilities_[
                column_index
            ]

            scores += probabilities[
                :,
                X_binned[:, column_index - 1],
                X_binned[:, column_index],
            ].T

        return scores

    def predict(self, X):
        """Predict the most probable class for each observation."""

        scores = self._joint_log_scores(X)

        winning_indices = np.argmax(
            scores,
            axis=1,
        )

        return self.classes_[winning_indices]

    def predict_proba(self, X):
        """Return normalized class probabilities."""

        scores = self._joint_log_scores(X)

        # Stable normalization in log space.
        scores -= scores.max(
            axis=1,
            keepdims=True,
        )

        probabilities = np.exp(scores)

        return probabilities / probabilities.sum(
            axis=1,
            keepdims=True,
        )