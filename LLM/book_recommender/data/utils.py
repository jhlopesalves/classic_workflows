import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate


def classification_evaluation(
	X: pd.DataFrame | ArrayLike,
	y: ArrayLike,
	estimator: BaseEstimator,
	*,
	cv: int = 5,
	threshold: float | None = None,
	random_state: int = 42,
) -> dict[str, float]:
	"""
	Evaluate binary classification using cross-validation.

	Parameters
	----------
	X : DataFrame or array-like
		Feature matrix.
	y : array-like
		True labels.
	estimator : BaseEstimator
		Unfitted scikit-learn estimator (cloned for each fold).
	cv : int, default=5
		Number of cross-validation folds.
	threshold : float or None, default=None
		Probability threshold for class assignment. If None, uses
		estimator.predict().
	random_state : int, default=42
		Random state for reproducibility.

	Returns
	-------
	dict[str, float]
		Dictionary with mean_auc, std_auc, mean_ap, std_ap.
	"""

	if not isinstance(y, (np.ndarray, pd.DataFrame, pd.Series, list)):
		raise TypeError(f"y must be array-like, got {type(y).__name__}")

	if not isinstance(X, (np.ndarray, pd.DataFrame)):
		raise TypeError(f"X must be either an Array or a DataFrame, got {type(X).__name__}")

	if not isinstance(estimator, BaseEstimator):
		raise TypeError(f"estimator must be a Scikit-Learn Estimator, got {type(estimator).__name__}")

	if threshold is not None:
		if not isinstance(threshold, (int, float)):
			raise TypeError(f"threshold must be a float in [0, 1] or None, got {type(threshold).__name__}")
		if not (0.0 <= float(threshold) <= 1.0):
			raise ValueError(f"threshold must be in [0, 1], got {threshold}")

	y_array = np.asarray(y)

	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

	# Initialize containers for CV results
	cms: list[np.ndarray] = []
	tprs: list[np.ndarray] = []
	aucs: list[float] = []
	precisions_list: list[np.ndarray] = []
	aps: list[float] = []

	# Common interpolation points
	mean_fpr = np.linspace(0, 1, 100)
	mean_recall = np.linspace(0, 1, 100)

	# Perform cross-validation
	skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

	# Track unknown categories across folds
	unknown_category_columns: set[int] = set()

	# Custom warning filter
	with warnings.catch_warnings(record=True) as caught_warnings:
		warnings.filterwarnings("always", category=UserWarning, module="sklearn")

		for train_idx, val_idx in skf.split(X, y_array):
			# Split data
			if isinstance(X, pd.DataFrame):
				X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
			else:
				X_train, X_val = X[train_idx], X[val_idx]

			y_train, y_val = y_array[train_idx], y_array[val_idx]

			# Clone and fit estimator
			fold_estimator = clone(estimator)
			fold_estimator.fit(X_train, y_train)

			# Probability scores (needed for ROC/PR, and for thresholded predictions)
			if not hasattr(fold_estimator, "predict_proba"):
				raise TypeError("This evaluator requires predict_proba(). Your estimator does not implement it.")

			y_proba = fold_estimator.predict_proba(X_val)[:, 1]

			# Class predictions
			if threshold is None:
				y_pred = fold_estimator.predict(X_val)
			else:
				y_pred = (y_proba >= float(threshold)).astype(int)

			# Confusion matrix (normalised)
			cm = confusion_matrix(y_true=y_val, y_pred=y_pred, normalize="true")
			cms.append(cm)

			# ROC curve
			fpr, tpr, _ = roc_curve(y_true=y_val, y_score=y_proba)
			roc_auc = roc_auc_score(y_true=y_val, y_score=y_proba)

			interp_tpr = np.interp(mean_fpr, fpr, tpr)
			interp_tpr[0] = 0.0
			tprs.append(interp_tpr)
			aucs.append(float(roc_auc))

			# Precision-Recall curve
			precision, recall, _ = precision_recall_curve(y_true=y_val, y_score=y_proba)
			avg_precision = average_precision_score(y_true=y_val, y_score=y_proba)

			# Interpolate precision
			interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
			precisions_list.append(interp_precision)
			aps.append(float(avg_precision))

		# Process warnings
		for warning in caught_warnings:
			message = str(warning.message).lower()
			if "unknown categories" in message:
				match = re.search(r"columns \[([^\]]+)\]", str(warning.message))
				if match:
					cols = match.group(1).split(",")
					unknown_category_columns.update([int(c.strip()) for c in cols])

	# Display custom warning if unknown categories were found
	if unknown_category_columns:
		cols_list = sorted(list(unknown_category_columns))
		print(f"Unknown categories found in columns {cols_list} will be encoded as zeros.")

	# Mean Confusion Matrix
	mean_cm = np.mean(cms, axis=0)
	std_cm = np.std(cms, axis=0)

	# Create annotation labels with mean ± std
	annot_labels = np.array([[f"{mean_cm[row, col]:.1%}\n(±{std_cm[row, col]:.1%})" for col in range(2)] for row in range(2)])

	cm_df = pd.DataFrame(
		data=mean_cm,
		index=["Actual Negative", "Actual Positive"],
		columns=["Pred Negative", "Pred Positive"],
	)

	sns.heatmap(
		data=cm_df,
		annot=annot_labels,
		fmt="",
		cmap="Blues",
		cbar_kws={"label": "Proportion"},
		ax=axes[0],
		vmin=0,
		vmax=1,
	)
	axes[0].set_title(f"Mean Confusion Matrix\n({cv}-Fold CV, Row-Normalised)", fontsize=11, pad=9)
	axes[0].set_xlabel("Predicted Label")
	axes[0].set_ylabel("True Label")

	# Mean ROC Curve
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	std_tpr = np.std(tprs, axis=0)
	mean_auc = float(np.mean(aucs))
	std_auc = float(np.std(aucs))

	axes[1].plot(
		mean_fpr,
		mean_tpr,
		color="darkorange",
		linewidth=2.5,
		label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
	)

	axes[1].fill_between(
		mean_fpr,
		np.maximum(mean_tpr - std_tpr, 0),
		np.minimum(mean_tpr + std_tpr, 1),
		color="darkorange",
		alpha=0.2,
		label="± 1 std. dev.",
	)

	axes[1].plot(
		[0, 1],
		[0, 1],
		color="navy",
		linestyle="--",
		alpha=0.5,
		label="Chance (AUC = 0.5)",
	)
	axes[1].set_title(f"Mean ROC Curve ({cv}-Fold CV)", fontsize=11, pad=9)
	axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
	axes[1].set_ylabel("True Positive Rate (Sensitivity)")
	axes[1].set_xlim([-0.02, 1.02])
	axes[1].set_ylim([-0.02, 1.02])
	axes[1].legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
	axes[1].grid(True, alpha=0.4, linewidth=0.4, color="grey")

	# Mean Precision-Recall Curve
	mean_precision = np.mean(precisions_list, axis=0)
	std_precision = np.std(precisions_list, axis=0)
	mean_ap = float(np.mean(aps))
	std_ap = float(np.std(aps))

	axes[2].plot(
		mean_recall,
		mean_precision,
		color="forestgreen",
		linewidth=2.5,
		label=f"Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})",
	)

	axes[2].fill_between(
		mean_recall,
		np.maximum(mean_precision - std_precision, 0),
		np.minimum(mean_precision + std_precision, 1),
		color="forestgreen",
		alpha=0.2,
		label="± 1 std. dev.",
	)

	# Baseline prevalence
	baseline = float(np.mean(y_array))
	axes[2].axhline(
		y=baseline,
		color="dimgrey",
		linestyle="--",
		linewidth=2,
		alpha=0.7,
		label=f"Baseline (Prevalence = {baseline:.3f})",
	)

	axes[2].set_title(f"Mean Precision-Recall Curve ({cv}-Fold CV)", fontsize=11, pad=9)
	axes[2].set_xlabel("Recall (Sensitivity)")
	axes[2].set_ylabel("Precision (PPV)")
	axes[2].set_xlim([-0.02, 1.02])
	axes[2].set_ylim([0.0, 1.05])
	axes[2].legend(loc="best", frameon=True, fancybox=True, shadow=True)
	axes[2].grid(True, alpha=0.4, linewidth=0.4, color="grey")

	# Extract model name
	if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
		model_name = estimator.named_steps["model"].__class__.__name__
	elif hasattr(estimator, "steps"):
		model_name = estimator.steps[-1][1].__class__.__name__
	else:
		model_name = estimator.__class__.__name__

	threshold_text = "" if threshold is None else f" | threshold={float(threshold):.2f}"
	plt.suptitle(
		f"Binary Classification: {model_name} ({cv}-Fold Cross-Validation){threshold_text}",
		fontsize=14,
		fontweight="bold",
	)
	plt.tight_layout()
	plt.show()

	return {
		"mean_auc": mean_auc,
		"std_auc": std_auc,
		"mean_ap": mean_ap,
		"std_ap": std_ap,
	}


def evaluate_candidates_cls(
	candidates: dict[str, BaseEstimator],
	X,
	y,
	*,
	n_splits: int = 5,
	sort_by: str = "test_roc_auc",
	n_jobs: int = -1,
	verbose: bool = True,
) -> pd.DataFrame:
	"""
	Evaluates multiple classification models using Stratified K-Fold cross-validation.

	Parameters
	----------
	candidates : dict
	    Dictionary mapping model names to sklearn estimators/pipelines.
	X : array-like
	    Feature matrix.
	y : array-like
	    Target vector (binary).
	n_splits : int, optional
	    Number of cross-validation folds, by default 5.
	sort_by : str, optional
	    Metric to sort the results by (e.g., 'test_roc_auc', 'test_f1'),
	    by default "test_roc_auc".
	n_jobs : int, optional
	    Number of jobs to run in parallel, by default -1.
	verbose : bool, optional
	    Whether to print progress messages, by default True.

	Returns
	-------
	pd.DataFrame
	    A DataFrame containing the mean and std of the specified metrics for each model.
	"""
	# Use StratifiedKFold for classification to maintain class balance in folds
	cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

	# Define classification metrics
	# Note: 'roc_auc' requires the model to support predict_proba or decision_function
	scoring = {
		"roc_auc": "roc_auc",
		"accuracy": "accuracy",
		"f1": "f1",
		"log_loss": "neg_log_loss",
	}

	results_list = []

	for name, model in candidates.items():
		if verbose:
			print(f"Evaluating {name}...")

		# cross_validate handles the splitting and scoring
		cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True)

		row = {"Model": name}
		for metric_key, scores in cv_results.items():
			# Skip timing metrics for the clean table
			if "time" in metric_key:
				continue

			# Determine the base metric name (e.g., "test_log_loss" -> "log_loss")
			# This helps look up if we need to flip the sign
			base_metric = metric_key.replace("test_", "").replace("train_", "")

			# Check if the original sklearn scorer string starts with "neg_"
			# If so, flip the sign to make it positive (e.g., Log Loss)
			scorer_name = scoring.get(base_metric, "")
			mean_score = scores.mean()

			if "neg_" in scorer_name:
				mean_score = -mean_score

			row[f"{metric_key} (mean)"] = mean_score
			row[f"{metric_key} (std)"] = scores.std()

		results_list.append(row)

	df = pd.DataFrame(results_list)

	# Dynamic sort column finding
	# Default to the passed sort_by, strictly looking for the (mean) column
	sort_col = next((c for c in df.columns if sort_by in c and "(mean)" in c), "test_roc_auc (mean)")

	# Determine sort order: usually higher is better, except for Log Loss
	ascending = "log_loss" in sort_by

	return df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
