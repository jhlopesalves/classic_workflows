import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate


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
