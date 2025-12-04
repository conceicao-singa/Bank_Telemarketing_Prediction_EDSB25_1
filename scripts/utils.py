import os
import pandas as pd
from sklearn.model_selection import GridSearchCV

def save_processed_splits(X_train, y_train, X_test, y_test, preprocessor, path: str):

    os.makedirs(path, exist_ok=True)

    feature_names = preprocessor.get_feature_names_out()

    # Convert sparse matrices to dense if needed
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    # Safety check
    if X_train.shape[1] != len(feature_names):
        raise ValueError(f"X_train has {X_train.shape[1]} columns, expected {len(feature_names)}")
    if X_test.shape[1] != len(feature_names):
        raise ValueError(f"X_test has {X_test.shape[1]} columns, expected {len(feature_names)}")

    # Convert arrays to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test, columns=feature_names)

    # Ensure y is a Series
    y_train_df = pd.Series(y_train, name="y").reset_index(drop=True)
    y_test_df  = pd.Series(y_test, name="y").reset_index(drop=True)

    # Concatenate features + target
    train_final = pd.concat([X_train_df, y_train_df], axis=1)
    test_final  = pd.concat([X_test_df, y_test_df], axis=1)

    # Save
    train_final.to_csv(os.path.join(path, "train_processed.csv"), index=False)
    test_final.to_csv(os.path.join(path, "test_processed.csv"), index=False)

    print(f"Processed splits saved to {path}")
    
      
import joblib

# -----------------------------
# 1. Utility function to save model
# -----------------------------
import os
import joblib

def save_models(models, folder_path):
    """
    Save all fitted models in a dictionary to the specified folder.

    Parameters
    ----------
    models : dict
        Dictionary of {model_name: fitted_model}.
    folder_path : str
        Path to the folder where models will be saved.
    """
    os.makedirs(folder_path, exist_ok=True)
    for name, model in models.items():
        file_path = os.path.join(folder_path, f"{name}.pkl")
        joblib.dump(model, file_path)
        print(f"✅ Saved {name} to {file_path}")


def load_model(model_name, folder_path):
    """
    Load a single model by name from the specified folder.

    Parameters
    ----------
    model_name : str
        The name of the model (without .pkl extension).
    folder_path : str
        Path to the folder containing saved models.

    Returns
    -------
    object
        The loaded model object.
    """
    file_path = os.path.join(folder_path, f"{model_name}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved model found at {file_path}")
    model = joblib.load(file_path)
    print(f"✅ Loaded {model_name} from {file_path}")
    return model
