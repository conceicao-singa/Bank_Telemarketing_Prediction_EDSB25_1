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


def clean_feature_names(feature_names, prefixes=("cat__", "num__")):
    return [name.replace(prefixes[0], "").replace(prefixes[1], "") for name in feature_names]
#-----------------------------  --------------------------------

# Visualization function to plot histograms for numerical columns

#----------------------------- -------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

def plot_numerical_histograms(df, numerical_cols, bins=30, figsize=(15, 10), palette="Set2"):
    """
    Plot histograms for all numerical columns in a DataFrame using Seaborn.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing your data.
    numerical_cols : list
        List of column names (numerical features) to plot.
    bins : int, optional
        Number of bins for the histogram (default=30).
    figsize : tuple, optional
        Figure size (default=(15, 10)).
    palette : str or list, optional
        Seaborn color palette (default="Set2").
    """
    sns.set(style="whitegrid", palette=palette)

    # Create subplots grid
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], bins=bins, kde=True, ax=axes[i], color=sns.color_palette(palette)[i % len(sns.color_palette(palette))])
        axes[i].set_title(col)

    # Remove unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    
#-------------------------------------------------------------
# Visualization function to analyze categorical features    
#-------------------------------------------------------------

def analyze_categorical_features(df, categorical_cols, figsize=(10, 5), palette="Set2"):
    """
    Analyze categorical features by printing value counts and plotting bar plots.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing your data.
    categorical_cols : list
        List of categorical column names to analyze.
    figsize : tuple, optional
        Figure size for each plot (default=(10, 5)).
    palette : str or list, optional
        Seaborn color palette (default="Set2").
    """
    sns.set(style="whitegrid", palette=palette)

    print("\nValue counts for categorical features:")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())

        plt.figure(figsize=figsize)
        sns.countplot(
            data=df,
            y=col,
            order=df[col].value_counts().index,
            palette=palette,
            hue=df[col]
        )
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()
        
#-------------------------------------------------------------
# Visualization function to plot correlation heatmap
#-------------------------------------------------------------

def plot_correlation_heatmap(corr_matrix, figsize=(10, 8), cmap="coolwarm", title="Correlation Matrix"):
    """
    Plot a correlation heatmap using Seaborn.

    Parameters
    ----------
    corr_matrix : pandas.DataFrame
        Correlation matrix (from df.corr()).
    figsize : tuple, optional
        Size of the figure (default=(10, 8)).
    cmap : str, optional
        Colormap for the heatmap (default="coolwarm").
    title : str, optional
        Title of the plot (default="Correlation Matrix").
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
#------------------------------------------------------------
# Visualization function to plot conversion rate bar chart
#------------------------------------------------------------    
    

def plot_conversion_by_category(data, category_col, rate_col="conversion_rate",
                                 figsize=(10, 6), order=None, rotation=45,
                                 title=None, palette="pastel", save=False, filename=None):
    """
    Plot a polished conversion rate bar chart using Seaborn.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing conversion rates.
    category_col : str
        Categorical column to plot on x-axis.
    rate_col : str, optional
        Column with conversion rates (default="conversion_rate").
    figsize : tuple, optional
        Size of the figure (default=(10, 6)).
    order : list, optional
        Custom order for categories (default=None).
    rotation : int, optional
        Rotation angle for x-axis labels (default=45).
    title : str, optional
        Plot title (default="Conversion Rate by <category_col>").
    palette : str or list, optional
        Seaborn color palette (default="pastel").
    save : bool, optional
        Whether to save the plot (default=False).
    filename : str, optional
        Filename to save the plot (if save=True).
    """
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=data, x=category_col, y=rate_col, order=order, palette=palette, edgecolor="gray", hue=data[category_col])
    ax.set_title(title if title else f"Conversion Rate by {category_col.capitalize()}", fontsize=14, weight="bold")
    ax.set_xlabel(category_col.replace("_", " ").capitalize(), fontsize=12)
    ax.set_ylabel("Conversion Rate", fontsize=12)
    plt.xticks(rotation=rotation, ha="right")
    sns.despine()
    plt.tight_layout()

    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
    
    
    
    
def compute_conversion_rate(df, category_col, target_col="y"):
    """
    Compute conversion rate per category.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data with categorical column and target.
    category_col : str
        Categorical column to group by.
    target_col : str
        Binary target column (default="target").

    Returns
    -------
    pandas.DataFrame
        DataFrame with category and conversion_rate.
    """
    return (
        df.groupby(category_col)[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: "conversion_rate"})
    )
