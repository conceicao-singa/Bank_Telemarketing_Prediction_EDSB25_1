from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid")  # global seaborn style

def evaluate_cv_test(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    print("\nTest Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics

# -----------------------------
# 2. Comprehensive model evaluation function



def evaluate_model(pipe, X_train, y_train, X_test, y_test, model_name="Model", threshold=0.5):
    print(f"\n==================== {model_name} — TRAINING vs TEST EVALUATION ====================\n")

    # -----------------------------
    # Predictions
    # -----------------------------
    y_proba_train = pipe.predict_proba(X_train)[:, 1]
    y_pred_train  = (y_proba_train >= threshold).astype(int)

    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test  = (y_proba_test >= threshold).astype(int)

    # -----------------------------
    # Confusion Matrices (Seaborn heatmaps)
    # -----------------------------
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test  = confusion_matrix(y_test, y_pred_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"], ax=axes[0])
    axes[0].set_title(f"{model_name} — Training Confusion Matrix @ {threshold}")

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"], ax=axes[1])
    axes[1].set_title(f"{model_name} — Test Confusion Matrix @ {threshold}")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics_train = {
        "accuracy": accuracy_score(y_train, y_pred_train),
        "precision": precision_score(y_train, y_pred_train, zero_division=0),
        "recall": recall_score(y_train, y_pred_train, zero_division=0),
        "f1": f1_score(y_train, y_pred_train, zero_division=0),
        "roc_auc": roc_auc_score(y_train, y_proba_train)
    }

    metrics_test = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "f1": f1_score(y_test, y_pred_test, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba_test)
    }

    print("Training Metrics")
    print("----------------")
    for k, v in metrics_train.items():
        print(f"{k.capitalize():<10}: {v:.4f}")

    print("\nTest Metrics")
    print("------------")
    for k, v in metrics_test.items():
        print(f"{k.capitalize():<10}: {v:.4f}")

    # -----------------------------
    # ROC Curve (Seaborn lineplot)
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=fpr, y=tpr, label=f"AUC = {metrics_test['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — Test ROC Curve")
    plt.legend()
    plt.show()

    # -----------------------------
    # Lift Plot (Seaborn lineplot)
    # -----------------------------
    def compute_lift(y_true, y_proba, n_bins=10):
        df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
        df = df.sort_values(by="y_proba", ascending=False).reset_index(drop=True)
        df["bucket"] = pd.qcut(df.index, n_bins, labels=False)
        lift = df.groupby("bucket")["y_true"].mean() / df["y_true"].mean()
        return lift

    lift_values = compute_lift(y_test, y_proba_test, n_bins=10)

    plt.figure(figsize=(7, 4))
    sns.lineplot(x=range(1, 11), y=lift_values.values, marker="o")
    plt.title(f"{model_name} — Lift Curve (Test Set)")
    plt.xlabel("Decile (Top Ranked Customers)")
    plt.ylabel("Lift over Average")
    plt.grid(True)
    plt.show()

    return {"train": metrics_train, "test": metrics_test}

# ----------------------------
# 3. Model comparison function
# ----------------------------  

def lift_at_k(y_true, y_score, k=0.10):
    df = pd.DataFrame({"y": np.array(y_true), "score": np.array(y_score)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    n_top = max(1, int(np.floor(k * len(df))))
    top_pos = df.loc[:n_top-1, "y"].sum()
    overall_rate = df["y"].sum() / len(df)
    if overall_rate == 0:
        return np.nan
    return (top_pos / n_top) / overall_rate


def compare_models(models, X_test, y_test, save_path):
    """
    Evaluate multiple models, display comparison table, and save CSV.
    """
    results = []
    for name, pipe in models.items():
        print(f"Evaluating {name} ...")
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            try:
                y_proba = pipe.decision_function(X_test)
            except Exception:
                y_proba = pipe.predict(X_test).astype(float)

        y_pred = (y_proba >= 0.45).astype(int) if y_proba.ndim == 1 else pipe.predict(X_test)

        res = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "lift@10%": lift_at_k(y_test, y_proba, 0.10),
            "lift@20%": lift_at_k(y_test, y_proba, 0.20)
        }
        results.append(res)

    comp_df = pd.DataFrame(results).set_index("model").sort_values("roc_auc", ascending=False)
    from IPython.display import display
    display(comp_df)

    comp_df.to_csv(save_path)
    print(f"✅ Saved model comparison to {save_path}")

    return comp_df
