import pandas as pd

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
    """
    Remove duplicate rows and return cleaned dataframe
    """
    before = df.shape[0]
    df_clean = df.drop_duplicates()
    after = df_clean.shape[0]

    print(f"Duplicates removed: {before - after}")
    return df_clean