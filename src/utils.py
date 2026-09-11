import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def build_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    ordinal_cols = ["Education"]
    education_order = [["Basic", "2n Cycle", "Graduation", "Master", "PhD"]]
    nominal_cols = ["Marital_Status"]

    numeric_cols = [col for col in train_df.columns if col not in ordinal_cols + nominal_cols]

    skewness = train_df[numeric_cols].skew()
    skewed_cols = skewness[skewness.abs() > 0.75].index.tolist()
    normal_cols = [col for col in numeric_cols if col not in skewed_cols]

    skewed_pipeline = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])

    normal_numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(categories=education_order)),
        ("scaler", StandardScaler()),
    ])

    nominal_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("skewed_numeric", skewed_pipeline, skewed_cols),
        ("normal_numeric", normal_numeric_pipeline, normal_cols),
        ("ordinal_cat", ordinal_pipeline, ordinal_cols),
        ("nominal_cat", nominal_pipeline, nominal_cols),
    ])

    return preprocessor


def evaluate_kmeans(X: np.ndarray, k_range, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(rows)


def plot_kmeans_metrics(metrics_df: pd.DataFrame, title_suffix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#4C72B0")
    axes[0].set_title(f"Elbow Method (Inertia) - {title_suffix}")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True)

    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="#55A868")
    axes[1].set_title(f"Silhouette Score - {title_suffix}")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score (higher = better)")
    axes[1].grid(True)

    axes[2].plot(metrics_df["k"], metrics_df["davies_bouldin"], marker="o", color="#C44E52")
    axes[2].set_title(f"Davies-Bouldin Index - {title_suffix}")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("DB Index (lower = better)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
