import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def plot_classifier_metrics(
	X: pd.DataFrame | pd.Series | ArrayLike,
	y: ArrayLike,
	estimator: BaseEstimator,
	*,
	cv: int = 5,
	threshold: float | None = None,
	random_state: int = 42,
) -> None:
	"""
	Evaluate binary classification using cross-validation.

	Automatically wraps non-probabilistic estimators (like LinearSVC)
	in CalibratedClassifierCV to enable ROC/PR plotting.

	Parameters
	----------
	X : DataFrame or array-like
	    Feature matrix.
	y : array-like
	    True labels (can be strings or numeric).
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

	# Type Checks
	if not isinstance(y, (np.ndarray, pd.DataFrame, pd.Series, list)):
		raise TypeError(f"y must be array-like, got {type(y).__name__}")
	if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
		raise TypeError(f"X must be an Array, DataFrame or Series, got {type(X).__name__}")
	if not isinstance(estimator, BaseEstimator):
		raise TypeError(f"estimator must be a Scikit-Learn Estimator, got {type(estimator).__name__}")
	if threshold is not None:
		if not isinstance(threshold, (int, float)) or not (0.0 <= float(threshold) <= 1.0):
			raise ValueError(f"threshold must be in [0, 1], got {threshold}")

	# Encode labels to 0/1 (required for ROC/PR curves)
	label_encoder = LabelEncoder()
	y_encoded = label_encoder.fit_transform(np.asarray(y))
	class_names = label_encoder.classes_  # e.g., ['Fiction', 'Nonfiction']
	pos_class_name = class_names[1]  # The class encoded as 1

	# Ensure Probabilities Exist
	model_to_plot = clone(estimator)
	final_estimator = model_to_plot.steps[-1][1] if isinstance(model_to_plot, Pipeline) else model_to_plot

	if not hasattr(final_estimator, "predict_proba"):
		print(f"Notice: {final_estimator.__class__.__name__} lacks 'predict_proba'. Wrapping in CalibratedClassifierCV.")
		model_to_plot = CalibratedClassifierCV(model_to_plot, method="isotonic", cv=3)

	# Plotting Setup
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

	cms: list[np.ndarray] = []
	tprs: list[np.ndarray] = []
	aucs: list[float] = []
	precisions_list: list[np.ndarray] = []
	aps: list[float] = []

	mean_fpr = np.linspace(0, 1, 100)
	mean_recall = np.linspace(0, 1, 100)

	skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
	unknown_category_columns: set[int] = set()

	with warnings.catch_warnings(record=True) as caught_warnings:
		warnings.filterwarnings("always", category=UserWarning, module="sklearn")

		for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
			# Handle indexing for DataFrame/Series vs numpy arrays
			if isinstance(X, (pd.DataFrame, pd.Series)):
				X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
			else:
				X_train, X_val = X[train_idx], X[val_idx]

			# Use ORIGINAL labels for training (model expects original label format)
			y_original = np.asarray(y)
			y_train_orig, y_val_orig = y_original[train_idx], y_original[val_idx]

			# Encoded labels for metrics
			y_val_enc = y_encoded[val_idx]

			# Clone and fit with original labels
			fold_estimator = clone(model_to_plot)
			fold_estimator.fit(X_train, y_train_orig)

			# Get probabilities - need to find which column corresponds to pos_label
			y_proba_all = fold_estimator.predict_proba(X_val)

			# Find index of positive class in estimator's classes_
			estimator_classes = fold_estimator.classes_
			pos_idx = np.where(estimator_classes == pos_class_name)[0][0]
			y_proba = y_proba_all[:, pos_idx]

			# Get predictions
			if threshold is None:
				y_pred_orig = fold_estimator.predict(X_val)
				y_pred = label_encoder.transform(y_pred_orig)
			else:
				y_pred = (y_proba >= float(threshold)).astype(int)

			# Confusion Matrix (using encoded labels)
			cm = confusion_matrix(y_true=y_val_enc, y_pred=y_pred, normalize="true")
			cms.append(cm)

			# ROC Curve (using encoded labels - now numeric!)
			fpr, tpr, _ = roc_curve(y_true=y_val_enc, y_score=y_proba)
			roc_auc = roc_auc_score(y_true=y_val_enc, y_score=y_proba)
			interp_tpr = np.interp(mean_fpr, fpr, tpr)
			interp_tpr[0] = 0.0
			tprs.append(interp_tpr)
			aucs.append(float(roc_auc))

			# Precision-Recall Curve
			precision, recall, _ = precision_recall_curve(y_true=y_val_enc, probas_pred=y_proba)
			avg_precision = average_precision_score(y_true=y_val_enc, y_score=y_proba)
			interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
			precisions_list.append(interp_precision)
			aps.append(float(avg_precision))

		# Check for unknown category warnings
		for warning in caught_warnings:
			message = str(warning.message).lower()
			if "unknown categories" in message:
				match = re.search(r"columns \[([^\]]+)\]", str(warning.message))
				if match:
					cols = match.group(1).split(",")
					unknown_category_columns.update([int(c.strip()) for c in cols])

	if unknown_category_columns:
		cols_list = sorted(list(unknown_category_columns))
		print(f"Warning: Unknown categories in columns {cols_list} encoded as zeros.")

	# === PLOTTING ===

	# 1. Confusion Matrix
	mean_cm = np.mean(cms, axis=0)
	std_cm = np.std(cms, axis=0)
	annot_labels = np.array([[f"{mean_cm[row, col]:.1%}\n(±{std_cm[row, col]:.1%})" for col in range(2)] for row in range(2)])
	cm_df = pd.DataFrame(mean_cm, index=[f"Actual: {class_names[0]}", f"Actual: {class_names[1]}"], columns=[f"Pred: {class_names[0]}", f"Pred: {class_names[1]}"])
	sns.heatmap(data=cm_df, annot=annot_labels, fmt="", cmap="Blues", ax=axes[0], vmin=0, vmax=1)
	axes[0].set_title(f"Mean Confusion Matrix\n({cv}-Fold CV)", fontsize=11)

	# 2. ROC Curve
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	std_tpr = np.std(tprs, axis=0)
	mean_auc = float(np.mean(aucs))
	std_auc = float(np.std(aucs))

	axes[1].plot(mean_fpr, mean_tpr, color="darkorange", lw=2, label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
	axes[1].fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="orange", alpha=0.2)
	axes[1].plot([0, 1], [0, 1], linestyle="--", color="navy", lw=1.5, label="Random Classifier")
	axes[1].set_xlabel("False Positive Rate")
	axes[1].set_ylabel("True Positive Rate")
	axes[1].set_title("Mean ROC Curve")
	axes[1].legend(loc="lower right")
	axes[1].set_xlim([0.0, 1.0])
	axes[1].set_ylim([0.0, 1.05])

	# 3. Precision-Recall Curve
	mean_precision = np.mean(precisions_list, axis=0)
	std_precision = np.std(precisions_list, axis=0)
	mean_ap = float(np.mean(aps))
	std_ap = float(np.std(aps))

	axes[2].plot(mean_recall, mean_precision, color="green", lw=2, label=f"Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})")
	axes[2].fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color="green", alpha=0.2)
	axes[2].set_xlabel("Recall")
	axes[2].set_ylabel("Precision")
	axes[2].set_title("Mean Precision-Recall Curve")
	axes[2].legend(loc="lower left")
	axes[2].set_xlim([0.0, 1.0])
	axes[2].set_ylim([0.0, 1.05])

	# Final Title
	if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
		model_name = estimator.named_steps["model"].__class__.__name__
	elif hasattr(estimator, "steps"):
		model_name = estimator.steps[-1][1].__class__.__name__
	else:
		model_name = estimator.__class__.__name__

	threshold_text = "" if threshold is None else f" | threshold={float(threshold):.2f}"
	plt.suptitle(
		f"Binary Classification: {model_name} ({cv}-Fold CV){threshold_text}",
		fontsize=14,
		fontweight="bold",
	)
	plt.tight_layout()
	plt.show()


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

	Automatically handles models that do not natively support probability estimates
	(e.g., LinearSVC, RidgeClassifier) by wrapping them in CalibratedClassifierCV.

	Parameters
	----------
	candidates : dict[str, BaseEstimator]
		Dictionary mapping model names to sklearn estimators/pipelines.
	X : array-like
		Feature matrix.
	y : array-like
		Target vector (binary).
	n_splits : int, optional
		Number of cross-validation folds, by default 5.
	sort_by : str, optional
		Metric to sort the results by (e.g., 'test_roc_auc', 'test_accuracy'),
		by default "test_roc_auc".
	n_jobs : int, optional
		Number of jobs to run in parallel, by default -1.
	verbose : bool, optional
		Whether to print progress messages, by default True.

	Returns
	-------
	pd.DataFrame
		A DataFrame containing the mean and std of the specified metrics.
	"""

	# 1. Setup Cross-Validation Strategy
	cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

	# 2. Define Metrics
	# Note: 'roc_auc' and 'neg_log_loss' strictly require probabilities.
	scoring = {
		"roc_auc": "roc_auc",
		"accuracy": "accuracy",
		"f1": "f1",
		"log_loss": "neg_log_loss",
	}

	results_list = []

	for name, estimator in candidates.items():
		if verbose:
			print(f"Evaluating {name}...")

		# 3. Probability Handling (The "Safety Valve")
		# We work on a clone to avoid mutating the user's original dictionary
		model_to_eval = clone(estimator)

		# Check if the model (or the final step of a pipeline) has predict_proba
		# If not, we wrap it in CalibratedClassifierCV to force probability output.
		final_estimator = model_to_eval.steps[-1][1] if isinstance(model_to_eval, Pipeline) else model_to_eval

		has_proba = hasattr(final_estimator, "predict_proba")

		# Edge case: SVC has the method but it might be False; LinearSVC never has it.
		# If it's strictly missing, we wrap it.
		if not has_proba:
			if verbose:
				print(f"  -> {name} lacks native 'predict_proba'. Wrapping in CalibratedClassifierCV (this may be slow).")
			# CalibratedClassifierCV uses its own internal CV to estimate probs
			model_to_eval = CalibratedClassifierCV(model_to_eval)

		# 4. Run Cross-Validation
		cv_results = cross_validate(model_to_eval, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True)

		# 5. Process Results
		row = {"Model": name}
		for metric_key, scores in cv_results.items():
			if "time" in metric_key:
				continue

			# Clean up metric names (e.g. "test_log_loss")
			base_metric = metric_key.replace("test_", "").replace("train_", "")

			# Flip negative scores (Log Loss) to be positive/readable
			scorer_name = scoring.get(base_metric, "")
			mean_score = scores.mean()

			if "neg_" in scorer_name:
				mean_score = -mean_score

			row[f"{metric_key} (mean)"] = mean_score
			row[f"{metric_key} (std)"] = scores.std()

		results_list.append(row)

	# 6. Formatting Output
	df = pd.DataFrame(results_list)

	# Locate the column to sort by (handling the "(mean)" suffix automatically)
	sort_col = next((c for c in df.columns if sort_by in c and "(mean)" in c), "test_roc_auc (mean)")

	# Log Loss is "lower is better", everything else is "higher is better"
	ascending = "log_loss" in sort_by

	return df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
