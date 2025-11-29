import os
import pandas as pd

def save_processed_splits(X_train, y_train, X_test, y_test, preprocessor, path: str):
    """
    Save processed train and test splits (features + target) as CSV files.

    Parameters:
        X_train (array-like): Processed training features
        y_train (pd.Series): Training target
        X_test (array-like): Processed test features
        y_test (pd.Series): Test target
        preprocessor (ColumnTransformer): Fitted preprocessor to extract feature names
        path (str): Directory path to save files
    """
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)

    # Get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Convert arrays to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test, columns=feature_names)

    # Reset index on y to align properly
    y_train_df = y_train.reset_index(drop=True)
    y_test_df  = y_test.reset_index(drop=True)

    # Concatenate features + target
    train_final = pd.concat([X_train_df, y_train_df], axis=1)
    test_final  = pd.concat([X_test_df, y_test_df], axis=1)

    # Save to CSV
    train_final.to_csv(os.path.join(path, "train_processed.csv"), index=False)
    test_final.to_csv(os.path.join(path, "test_processed.csv"), index=False)

    print(f"Processed splits saved to {path}")