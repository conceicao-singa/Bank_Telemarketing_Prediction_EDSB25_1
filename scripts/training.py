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
    
    """""""""""""""""""""""""""""""""""""""""""""""
    Decision Trees Training and Evaluation Modules  
    """""""""""""""""""""""""""""""""""""""""""""""
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np

def build_tree_pipeline(random_state: int = 42):
    """Create a pipeline with SMOTE and Decision Tree."""
    pipe = Pipeline([
        ("smote", SMOTE(sampling_strategy='minority', random_state=random_state)),
        ("model", DecisionTreeClassifier(random_state=random_state))
    ])
    return pipe



def train_and_evaluate_tree(pipe, X_train, y_train, X_test, y_test, threshold: float = 0.5):
    
    """Train Decision Tree pipeline and evaluate on test set."""
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    print("\nTest Results:")
    for k, v in metrics.items():
        print(f"{k.capitalize():<10}: {v:.4f}")

    return metrics

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Random Forest Training and Evaluation Modules
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.ensemble import RandomForestClassifier
def build_rf_pipeline(n_estimators=200, random_state=42):
    return Pipeline([
        ("smote", SMOTE(sampling_strategy="minority", random_state=random_state)),
        ("model", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
    ])

# -----------------------------
# 3. Test set evaluation
# -----------------------------
def evaluate_test(pipeline, X_train, y_train, X_test, y_test):
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


    """""""""""""""""""""""""""""""""""""""""""""""
    Support Vector Machine and Evaluation Modules  
    """""""""""""""""""""""""""""""""""""""""""""""
from sklearn.svm import SVC

# -----------------------------
# 1. Build pipeline
# -----------------------------
def build_svm_pipeline(kernel="rbf", probability=True, random_state=42):
    return Pipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("model", SVC(kernel=kernel, probability=probability, random_state=random_state))
    ])


    """""""""""""""""""""""""""""""""""""""""""""""
    Neural networks  Modules  
    """""""""""""""""""""""""""""""""""""""""""""""
    
from sklearn.neural_network import MLPClassifier

# -----------------------------
# 1. Build pipeline
# -----------------------------

def build_mlp_pipeline(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42):
    return Pipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("model", MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                max_iter=max_iter,
                                random_state=random_state))
    ])

    """""""""""""""""""""""""""""""""""""""""""""""
    XGBoost Modules  
    """""""""""""""""""""""""""""""""""""""""""""""
from xgboost import XGBClassifier


def build_xgb_pipeline(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
):
    return Pipeline([
        ("smote", SMOTE(sampling_strategy="minority", random_state=random_state)),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=1,   # balanced via SMOTE
            random_state=random_state,
            n_jobs=-1
        ))
    ])
