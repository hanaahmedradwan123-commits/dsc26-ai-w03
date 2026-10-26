import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Target column name
TARGET = "failed"

# Explicit list of raw features allowed at creation time (excludes parcel_id, customer_id, courier_notes, delivered_at, failed)
FEATURE_COLUMNS = [
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes point-in-time customer history features without data leakage."""
    # Work on a copy to ensure input is never mutated
    result = df.copy()

    # Pre-allocate feature lists maintaining original index order
    n = len(result)
    prior_parcels = [0] * n
    prior_fail_rate = [np.nan] * n

    # Group by customer_id to extract historical features
    for customer_id, group in result.groupby("customer_id", sort=False):
        group_indices = group.index.tolist()
        group_times = group["created_at"].values
        group_failed = group["failed"].values

        for idx, t_curr in zip(group_indices, group_times):
            # Strict point-in-time: strictly earlier than current created_at timestamp
            mask = group_times < t_curr
            count = int(np.sum(mask))
            prior_parcels[idx] = count

            if count > 0:
                fails = np.sum(group_failed[mask])
                prior_fail_rate[idx] = float(fails / count)

    result["customer_prior_parcels"] = prior_parcels
    result["customer_prior_fail_rate"] = prior_fail_rate

    return result


def time_split(df: pd.DataFrame, time_col: str = "created_at", test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data temporally with boundary timestamp rule (all boundary rows to test)."""
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    n_total = len(df)
    n_test = math.ceil(n_total * test_fraction)

    # Sort strictly by time using stable sort
    sorted_df = df.sort_values(by=time_col, kind="mergesort")

    boundary_idx = n_total - n_test
    boundary_timestamp = sorted_df.iloc[boundary_idx][time_col]

    # Rule: All rows sharing the boundary timestamp go to test
    train_mask = df[time_col] < boundary_timestamp
    test_mask = df[time_col] >= boundary_timestamp

    train_df = df[train_mask]
    test_df = df[test_mask]

    return train_df, test_df


class TextLengthTransformer(BaseEstimator, TransformerMixin):
    """Transformer for numeric extraction of address length."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Safely extract values regardless of whether X is a DataFrame or 2D NumPy array
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        else:
            series = pd.Series(np.asarray(X).ravel())

        lengths = series.fillna("").astype(str).str.len().values
        return lengths.reshape(-1, 1)


def build_pipeline(seed: int = 0) -> Pipeline:
    """Returns an unfitted sklearn Pipeline supporting missing categories and text."""
    num_cols = ["cod_amount", "n_items", "customer_prior_parcels", "customer_prior_fail_rate"]
    cat_cols = ["merchant_id", "area", "promised_slot"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    address_pipeline = Pipeline([
        ("length", TextLengthTransformer()),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            ("address", address_pipeline, ["address_text"]),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(random_state=seed, max_iter=1000)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Evaluates fitted model on test set using roc_auc, pr_auc, brier, base_rate, n."""
    y_true = test_df[TARGET].values
    y_prob = model.predict_proba(test_df)[:, 1]

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    base_rate = float(np.mean(y_true))
    n = len(y_true)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "base_rate": base_rate,
        "n": n,
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    """Fits and compares model pipeline against DummyClassifier baseline."""
    # 1. Fit & evaluate model
    model = build_pipeline(seed=seed)
    model.fit(train_df, train_df[TARGET])
    model_metrics = evaluate(model, test_df)

    # 2. Fit & evaluate dummy baseline
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_df, train_df[TARGET])
    dummy_metrics = evaluate(dummy, test_df)

    lift_pr_auc = float(model_metrics["pr_auc"] - dummy_metrics["pr_auc"])

    return {
        "model": model_metrics,
        "dummy": dummy_metrics,
        "lift_pr_auc": lift_pr_auc,
    }