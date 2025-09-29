import os
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------- I/O ----------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and perform basic cleaning used in your notebook."""
    df = pd.read_csv(csv_path)
    # Map labels (M->1, B->0)
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})
    # Drop known non-features if present
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    return df

# ---------- Feature Engineering ----------

def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.select_dtypes(include=[np.number]).drop(columns=["diagnosis"], errors="ignore")
    y = df["diagnosis"]
    return X, y

def select_k_best(X: pd.DataFrame, y: pd.Series, k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()].tolist()
    X_sel = pd.DataFrame(X_new, columns=selected_cols, index=X.index)
    return X_sel, selected_cols

def apply_scaler(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return Xtr, Xte, scaler

def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    pca = PCA(n_components=min(n_components, X_train.shape[1]))
    Xtr = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index)
    Xte = pd.DataFrame(pca.transform(X_test), index=X_test.index)
    return Xtr, Xte, pca

# ---------- Metrics & Reporting ----------

def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 3),
        "precision": round(precision_score(y_true, y_pred) * 100, 3),
        "recall": round(recall_score(y_true, y_pred) * 100, 3),
        "f1": round(f1_score(y_true, y_pred) * 100, 3),
    }

def save_confusion_matrix_plot(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_bar_comparison(results_df: pd.DataFrame, out_path: str, title: str = "Model Performance Comparison") -> None:
    import matplotlib.pyplot as plt
    df_plot = results_df.set_index("Model")
    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------- Persistence ----------

def save_artifacts(
    save_dir: str,
    model,
    scaler: Optional[StandardScaler] = None,
    pca: Optional[PCA] = None,
    model_name: str = "model"
) -> None:
    """Save model + optional scaler/pca with joblib."""
    ensure_dir(save_dir)
    import joblib
    joblib.dump(model, os.path.join(save_dir, f"{model_name}.joblib"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    if pca is not None:
        joblib.dump(pca, os.path.join(save_dir, "pca.joblib"))
