import pandas as pd
import numpy as np

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace('.', '_') for c in df.columns]
    return df

def encode_target(df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame:
    df[target_col] = (df[target_col].str.lower().str.strip() == 'yes').astype(int)
    return df

def split_features(df: pd.DataFrame):
    categoricals = ["job","marital","education","default","housing",
                    "loan","contact","month","day_of_week","poutcome"]
    numericals = ["age","campaign","pdays","previous","emp_var_rate",
                  "cons_price_idx","cons_conf_idx","euribor3m","nr_employed"]

    X_act = df[categoricals + numericals].copy()
    X_bmk = df[categoricals + ["duration"] + numericals].copy()
    y = df["y"].values
    return X_act, X_bmk, y

def create_actionable_df(X_act: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    df_act = X_act.assign(y=y)
    df_act = df_act.drop_duplicates()
    return df_act

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and return cleaned dataframe"""
    before = df.shape[0]
    df_clean = df.drop_duplicates()
    after = df_clean.shape[0]
    print(f"Duplicates removed: {before - after}")
    return df_clean


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Create an 'age_group' feature by binning age into categories."""
    bins = [0, 25, 50, 75, 100]
    labels = ['youth', 'adult', 'senior', 'elderly']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df


def add_housing_loan_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Create an interaction term between housing and loan features."""
    df['housing_loan_interaction'] = df['housing'] + '_' + df['loan']
    return df


def impute_marital_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'unknown' values in marital column with the mode."""
    mode_val = df['marital'].mode().iloc[0]
    df['marital'] = df['marital'].replace('unknown', mode_val)
    return df


def transform_pdays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 999 in 'pdays' with 'never_contacted' and bin remaining values.
    Creates a categorical feature 'pdays_cat' and drops original 'pdays'.
    """
    df['pdays_cat'] = df['pdays']
    df['pdays_cat'] = df['pdays_cat'].replace(999, 'never_contacted')

    mask = df['pdays_cat'] != 'never_contacted'
    df.loc[mask, 'pdays_cat'] = pd.cut(
        df.loc[mask, 'pdays_cat'].astype(int),
        bins=[-1, 3, 7, 15, np.inf],
        labels=['0-3', '4-7', '8-15', '16+']
    )

    df['pdays_cat'] = df['pdays_cat'].astype('category')
    df = df.drop(columns=['pdays'])
    return df


def get_features_to_keep(df: pd.DataFrame, features_to_drop=None) -> list:
    """
    Return list of features to keep, excluding specified features and target 'y'.
    """
    if features_to_drop is None:
        features_to_drop = ['default']
    features_to_keep = [col for col in df.columns if col not in features_to_drop + ['y']]
    return features_to_keep




from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_features_target(df: pd.DataFrame, target_col: str = "y"):
    """Split dataframe into features (X) and target (y)."""
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y


def get_column_types(X: pd.DataFrame):
    """Identify numerical and categorical columns."""
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    """Create a ColumnTransformer with scaling and one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor


def time_based_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8):
    """Perform a time-based split of features and target."""
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test, preprocessor):
    """Fit preprocessor on training data and transform both train and test sets."""
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed

## Outlier Removal Function Using IQR Method

def remove_outliers_iqr(df: pd.DataFrame, column: str, lower_quantile: float = 0.10, upper_quantile: float = 0.90) -> pd.DataFrame:
    Q1 = df[column].quantile(lower_quantile)
    Q3 = df[column].quantile(upper_quantile)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column}: Removed {outliers.shape[0]} outliers")

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

