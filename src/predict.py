# src/predict.py
import argparse, os, json, joblib, pandas as pd

def load_artifacts(models_dir: str):
    model_path = [p for p in os.listdir(models_dir)
                  if p.endswith(".joblib") and p not in ("scaler.joblib","pca.joblib")][0]
    model = joblib.load(os.path.join(models_dir, model_path))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    pca_path = os.path.join(models_dir, "pca.joblib")
    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None

    with open(os.path.join(models_dir, "run_summary.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    used_features = meta["used_features"]  # the canonical feature list/order
    return model, scaler, pca, used_features

def align_features(df: pd.DataFrame, used_features):
    # drop known non-features if present
    df = df.drop(columns=["diagnosis","id","Unnamed: 32"], errors="ignore")
    # keep numeric only
    df = df.select_dtypes(include="number")

    missing = [c for c in used_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required feature(s): {missing}\n"
            f"Your CSV must contain exactly these columns (any order is fine):\n{used_features}"
        )
    # select and ORDER exactly like during fit
    df = df[used_features]
    return df

def predict_df(df: pd.DataFrame, model, scaler, pca, used_features):
    X = align_features(df, used_features)
    Xs = scaler.transform(X)
    if pca is not None:
        Xs = pca.transform(Xs)
    return model.predict(Xs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=str, default="./models/run1")
    ap.add_argument("--csv", type=str, required=True,
                    help="CSV with feature columns (no diagnosis)")
    args = ap.parse_args()

    model, scaler, pca, used_features = load_artifacts(args.models_dir)
    df = pd.read_csv(args.csv)
    preds = predict_df(df, model, scaler, pca, used_features)
    print(preds.tolist())  # 0 = Benign, 1 = Malignant
