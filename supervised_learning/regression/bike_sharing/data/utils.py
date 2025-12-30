import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


def summarise_categorical(
    df: pd.DataFrame,
    column: str,
    target: str = "log_cnt",
) -> pd.DataFrame:
    group = df.groupby(column, observed=True)[target].agg(["count", "mean", "median", "std"])

    # Add proportion of total rows
    total_count = len(df)
    group["proportion"] = group["count"] / total_count

    # Optional: sort by mean or by count, your call
    summary = group.sort_values("mean", ascending=False)

    return summary


def evaluate_candidates(
    candidates: dict[str, BaseEstimator],
    X,
    y,
    *,
    n_splits: int = 5,
    sort_by: str = "test_rmse",
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    results_list = []

    for name, model in candidates.items():
        if verbose:
            print(f"Evaluating {name}...")
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True)

        # Aggregate logic
        row = {"Model": name}
        for metric_name, scores in cv_results.items():
            if "time" in metric_name:
                continue
            # Flip negative scores
            if "neg_" in scoring.get(metric_name.replace("test_", "").replace("train_", ""), ""):
                scores = -scores
            row[f"{metric_name} (mean)"] = scores.mean()
            row[f"{metric_name} (std)"] = scores.std()
        results_list.append(row)

    df = pd.DataFrame(results_list)
    # Dynamic sort column finding
    sort_col = next((c for c in df.columns if sort_by in c and "(mean)" in c), "test_rmse (mean)")
    return df.sort_values(sort_col, ascending=("r2" not in sort_by)).reset_index(drop=True)


def plot_learning_curve(
    estimator,
    X,
    y,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    title="Learning Curve",
):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        shuffle=True,
        random_state=42,
        return_times=False,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training curve
    ax.plot(
        train_sizes,
        train_mean,
        label="Training Score",
        color="tab:blue",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="tab:blue",
    )

    # Plot cross-validation curve
    ax.plot(
        train_sizes,
        test_mean,
        label="Cross-Validation Score",
        color="tab:orange",
        linewidth=2.5,
        marker="s",
        markersize=6,
    )
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.15,
        color="tab:orange",
    )

    # Styling
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Training Set Size", fontsize=11)
    ax.set_ylabel(scoring.upper() if scoring in ["r2", "mae", "mse"] else scoring.replace("_", " ").title(), fontsize=11)
    ax.legend(loc="best", frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.4, linewidth=0.5, color="grey")
    ax.set_axisbelow(True)

    # Add minor gridlines for extra polish
    ax.minorticks_on()
    ax.grid(which="minor", alpha=0.2, linewidth=0.3, color="grey", linestyle=":")

    plt.tight_layout()
    plt.show()


class IntegerDecoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping: dict, unknown_label: str | int = "Unknown"):
        self.mapping = mapping
        self.unknown_label = unknown_label

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IntegerDecoder expects a pandas DataFrame.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IntegerDecoder expects a pandas DataFrame.")

        X = X.copy()

        for col, code_map in self.mapping.items():
            if col in X.columns:
                # 1. Perform the mapping
                mapped_series = X[col].map(code_map)

                # 2. Fill unknowns
                mapped_series = mapped_series.fillna(self.unknown_label)

                # 3. Assign back
                X[col] = mapped_series

        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
            return self.feature_names_in_
        if len(input_features) != self.n_features_in_:
            raise ValueError("input_features must have the same length as feature_names_in_")
        return np.asarray(input_features, dtype=object)


def make_mapping_pipe(*, mapping: dict, transformer: type[BaseEstimator], drop_first: bool = True, sparse_output: bool = False) -> Pipeline:
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected 'mapping' to be a dict, got {type(mapping).__name__}")

    if not (isinstance(transformer, type) and issubclass(transformer, BaseEstimator)):
        raise TypeError(f"Expected 'transformer' to be a BaseEstimator class, got {type(transformer).__name__}")

    return Pipeline(
        steps=[
            ("mapping", transformer(mapping=mapping)),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", drop="first" if drop_first else None, sparse_output=sparse_output)),
        ]
    )


def make_numerical_pipe(*, transformer: TransformerMixin | None = None) -> Pipeline:
    if transformer is None:
        transformer = StandardScaler()
    if not isinstance(transformer, BaseEstimator):
        raise TypeError(f"Expected 'transformer' to be a BaseEstimator, got {type(transformer).__name__}")
    return Pipeline(steps=[("num_scaler", transformer)])


def make_categorical_pipe(*, drop_first: bool = True, sparse_output: bool = False) -> Pipeline:
    return Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", drop="first" if drop_first else None, sparse_output=sparse_output))])


def make_preprocessor(
    *, categorical_cols: list[str], numerical_cols: list[str], mapping: dict, mapping_cols: list[str], numerical_transformer: TransformerMixin | None = None
) -> ColumnTransformer:

    transformers = [
        ("categorical", make_categorical_pipe(), categorical_cols),
        ("numerical", make_numerical_pipe(), numerical_cols),
        ("mapping", make_mapping_pipe(mapping=mapping, transformer=IntegerDecoder), mapping_cols),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_full_pipeline(
    *,
    model: BaseEstimator | None = None,
    categorical_cols: list | None = None,
    numerical_cols: list,
    mapping_cols: list,
    mapping: dict,
    log_target: bool = True,
) -> BaseEstimator:

    if model is None:
        model = LinearRegression()

    if categorical_cols is None:
        categorical_cols = []

    preprocessor = make_preprocessor(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        mapping_cols=mapping_cols,
        mapping=mapping,
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ]
    )

    if log_target:
        return TransformedTargetRegressor(regressor=full_pipeline, func=np.log1p, inverse_func=np.expm1)

    return full_pipeline


if __name__ == "__main__":
    # Quick smoke test to verify imports and basic functionality
    from sklearn.datasets import make_regression

    print("Testing utils module...")

    # Test data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f"num_{i}" for i in range(5)])

    # Test make_full_pipeline
    pipeline = make_full_pipeline(
        numerical_cols=X.columns.tolist(),
        mapping_cols=[],
        mapping={},
        categorical_cols=[],
        log_target=False,
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    print(f"Pipeline fitted successfully. Sample predictions: {predictions[:3]}")
    print("All tests passed!")
