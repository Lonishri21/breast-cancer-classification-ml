import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from utils import (
    ensure_dir, load_data, split_X_y, select_k_best,
    apply_scaler, apply_pca, metrics_dict,
    save_confusion_matrix_plot, save_bar_comparison,
    save_text, save_json, save_artifacts
)

def build_models(choice: str):
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(kernel="rbf", probability=False, random_state=42),
    }
    if choice == "all":
        return models
    else:
        return {choice: models[choice]}

def main(args):
    # Output dirs
    reports_dir = args.reports_dir
    ensure_dir(reports_dir)
    if args.save_dir:
        ensure_dir(args.save_dir)

    # 1) Load & split
    df = load_data(args.data)
    X, y = split_X_y(df)

    # Optional feature selection
    selected_cols = None
    if args.kbest and args.kbest > 0:
        X, selected_cols = select_k_best(X, y, k=args.kbest)

    used_features = X.columns.tolist()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # Scale
    X_train_s, X_test_s, scaler = apply_scaler(X_train, X_test)

    # Optional PCA
    pca_obj = None
    if args.pca and args.pca > 0:
        X_train_s, X_test_s, pca_obj = apply_pca(X_train_s, X_test_s, n_components=args.pca)

    # 2) Build model(s)
    models = build_models(args.model)

    # 3) Train & evaluate
    rows = []
    best_name, best_f1, best_model = None, -1, None
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        m = metrics_dict(y_test, y_pred)
        rows.append([name, m["accuracy"], m["precision"], m["recall"], m["f1"]])

        # Save confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix_plot(
            cm, os.path.join(reports_dir, f"cm_{name}.png"),
            title=f"Confusion Matrix: {name}"
        )

        # Save classification report
        cr = classification_report(y_test, y_pred, target_names=["Benign","Malignant"])
        save_text(cr, os.path.join(reports_dir, f"classification_report_{name}.txt"))

        # Track best by F1
        if m["f1"] > best_f1:
            best_f1, best_name, best_model = m["f1"], name, model

    # 4) Persist best artifacts
    if best_model is not None and args.save_dir:
        meta = {
            "best_model": best_name,
            "f1": best_f1,
            "selected_features": selected_cols,
            "used_features": used_features,
            "seed": args.seed,
            "test_size": args.test_size
        }
        save_json(meta, os.path.join(args.save_dir, "run_summary.json"))
        save_artifacts(args.save_dir, best_model, scaler=scaler, pca=pca_obj, model_name=best_name)

    # 5) Comparison table & plot
    results_df = pd.DataFrame(rows, columns=["Model", "Accuracy(%)", "Precision(%)", "Recall(%)", "F1(%)"])
    results_df.sort_values("F1(%)", ascending=False, inplace=True)
    results_df.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    save_bar_comparison(results_df, os.path.join(reports_dir, "model_comparison.png"))

    # Console summary
    print("\n=== Model Comparison (sorted by F1) ===")
    print(results_df.to_string(index=False))
    if args.save_dir:
        print(f"\nBest model saved to: {args.save_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train breast cancer classifiers.")
    p.add_argument("--data", type=str, default="./data.csv", help="Path to data.csv")
    p.add_argument("--model", type=str, default="all", choices=["all","logreg","rf","svm"], help="Which model(s) to train")
    p.add_argument("--kbest", type=int, default=10, help="Top-K features via ANOVA F-test (0 = disable)")
    p.add_argument("--pca", type=int, default=0, help="PCA components after scaling (0 = disable)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save-dir", type=str, default="./models", help="Directory to save best model & scaler/PCA")
    p.add_argument("--reports-dir", type=str, default="./reports", help="Directory to save plots & metrics")
    args = p.parse_args()
    main(args)
