from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np



"""""""""""""""""""""""""""""""""""""""""""""""""""
Logistic Regression Training and Evaluation Modules
"""""""""""""""""""""""""""""""""""""""""""""""""""



def build_logreg_pipeline(random_state: int = 42):
    """Create a pipeline with SMOTE and logistic regression."""
    pipe = Pipeline([
        ("smote", SMOTE(sampling_strategy='minority', random_state=random_state)),
        ("model", LogisticRegression(solver="liblinear", class_weight="balanced"))
    ])
    return pipe


def evaluate_pipeline_cv(pipe, X, y, n_splits: int = 5, scoring: str = "roc_auc"):
    """Run cross-validation and return fold scores and mean score."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = cross_validate(
        pipe,
        X,
        y,
        cv=tscv,
        scoring=scoring,
        return_train_score=False
    )
    fold_scores = cv_results["test_score"]
    mean_score = np.mean(fold_scores)

    print("ROC-AUC for each fold:")
    for i, score in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"\nMean CV ROC-AUC: {mean_score:.4f}")

    return fold_scores, mean_score


def train_and_evaluate(pipe, X_train, y_train, X_test, y_test):
    """Train pipeline and evaluate on test set."""
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print("\nTest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def run_grid_search(pipe, param_grid, X_train, y_train, X_test, y_test, cv, scoring="roc_auc"):
    """
    Run GridSearchCV on a pipeline and evaluate best model on test set.
    """
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    print("Best Params:", gs.best_params_)
    print("Best CV ROC-AUC:", gs.best_score_)

    # Evaluate on test set
    y_proba = gs.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\nTest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    return gs.best_params_, gs.best_score_, {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }