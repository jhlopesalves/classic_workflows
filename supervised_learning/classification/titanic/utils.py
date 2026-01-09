from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

MetricName = Literal["f1", "recall", "precision", "accuracy"]


def oof_predict_proba_binary(
	estimator: BaseEstimator,
	X: pd.DataFrame | np.ndarray,
	y: ArrayLike,
	*,
	n_splits: int = 5,
	random_state: int = 42,
) -> np.ndarray:
	y_array = np.asarray(y).astype(int)
	proba_oof = np.zeros_like(y_array, dtype=float)

	cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

	for train_idx, val_idx in cv.split(X, y_array):
		if isinstance(X, pd.DataFrame):
			X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
		else:
			X_train, X_val = X[train_idx], X[val_idx]

		y_train = y_array[train_idx]

		fold_estimator = clone(estimator)
		fold_estimator.fit(X_train, y_train)

		proba_oof[val_idx] = fold_estimator.predict_proba(X_val)[:, 1]

	return proba_oof


@dataclass(frozen=True)
class ThresholdResult:
	threshold: float
	f1: float
	recall: float
	precision: float
	accuracy: float


def threshold_sweep(
	y_true: ArrayLike,
	proba: ArrayLike,
	*,
	thresholds: np.ndarray | None = None,
) -> list[ThresholdResult]:
	y_true_array = np.asarray(y_true).astype(int)
	proba_array = np.asarray(proba).astype(float)

	if thresholds is None:
		thresholds = np.linspace(0.05, 0.95, 181)

	results: list[ThresholdResult] = []
	for threshold in thresholds:
		y_pred = (proba_array >= threshold).astype(int)

		results.append(
			ThresholdResult(
				threshold=float(threshold),
				f1=float(f1_score(y_true_array, y_pred, zero_division=0)),
				recall=float(recall_score(y_true_array, y_pred, zero_division=0)),
				precision=float(precision_score(y_true_array, y_pred, zero_division=0)),
				accuracy=float(accuracy_score(y_true_array, y_pred)),
			)
		)

	return results


def pick_best_threshold(
	results: list[ThresholdResult],
	*,
	optimise: MetricName = "f1",
) -> ThresholdResult:
	key_map = {
		"f1": lambda r: r.f1,
		"recall": lambda r: r.recall,
		"precision": lambda r: r.precision,
		"accuracy": lambda r: r.accuracy,
	}
	key_fn = key_map[optimise]
	return max(results, key=key_fn)
