import os
import pandas as pd

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
    y_train_df = pd.Series(y_train, name="target").reset_index(drop=True)
    y_test_df  = pd.Series(y_test, name="target").reset_index(drop=True)

    # Concatenate features + target
    train_final = pd.concat([X_train_df, y_train_df], axis=1)
    test_final  = pd.concat([X_test_df, y_test_df], axis=1)

    # Save
    train_final.to_csv(os.path.join(path, "train_processed.csv"), index=False)
    test_final.to_csv(os.path.join(path, "test_processed.csv"), index=False)

    print(f"Processed splits saved to {path}")