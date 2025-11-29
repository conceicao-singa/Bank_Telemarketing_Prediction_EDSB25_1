from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")  # global seaborn style

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

    return {"train": metrics_train, "test": metrics_test}