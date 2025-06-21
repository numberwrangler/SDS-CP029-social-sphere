"""
Utility functions for social media conflict classification project.

This module contains reusable functions for data preprocessing, model training,
evaluation, and visualization used in the classification notebooks.
"""

# Standard library imports
import json
import os

# Third-party imports
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models.signature import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
from scipy import sparse
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ======== Warnings ========
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column"
)
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")



# ================== Data preprocessing ==================
def create_binary_conflict(df, target_column='Conflicts', threshold=None, visualize=True):
    """
    Create binary conflict classification (High vs Low) from conflict scores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, default='Conflicts'
        Name of the conflict column
    threshold : int or float, optional
        Threshold for binary classification. If None, uses median.
    visualize : bool, default=True
        Whether to create visualizations
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added binary conflict column
    dict
        Dictionary with analysis results including threshold, counts, and imbalance ratio
    """
    # Make a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Determine threshold
    if threshold is None:
        threshold = df_copy[target_column].median()
    
    # Create binary target variable
    df_copy['Conflict_Binary'] = df_copy[target_column].apply(
        lambda x: 'High' if x > threshold else 'Low'
    )
    
    # Calculate statistics
    conflict_counts = df_copy['Conflict_Binary'].value_counts()
    imbalance_ratio = conflict_counts.min() / conflict_counts.max() * 100
    
    # Print analysis
    print(f"Binary Conflict Classification:")
    print(f"Threshold: {threshold}")
    print(f"Low Conflict (0-{threshold}): {conflict_counts.get('Low', 0)} samples")
    print(f"High Conflict ({threshold+1}-max): {conflict_counts.get('High', 0)} samples")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}%")
    print(f"Class proportions:")
    print(conflict_counts / len(df_copy))

        # convert to 0 and 1
    df_copy['Conflict_Binary'] = df_copy['Conflict_Binary'].map({'Low': 0, 'High': 1})
    # print the first 5 rows of the binary variable
    # print(df_copy['Conflict_Binary'].head())
    
    # Create visualizations if requested
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Original distribution with threshold line
        axes[0].hist(df_copy[target_column], bins=range(int(df_copy[target_column].min()), 
                                                       int(df_copy[target_column].max()) + 2), 
                    alpha=0.7, edgecolor='black')
        axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold = {threshold}')
        axes[0].set_xlabel('Conflict Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Original Conflict Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Binary distribution
        sns.countplot(data=df_copy, x='Conflict_Binary', ax=axes[1])
        axes[1].set_title(f'Binary Conflict Distribution\n(Threshold: {threshold})\n(0: Low, 1: High)')
        axes[1].set_ylabel('Count')
        
        # Add count labels on bars
        for i, bar in enumerate(axes[1].patches):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    # Prepare results dictionary
    results = {
        'threshold': threshold,
        'counts': conflict_counts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'proportions': (conflict_counts / len(df_copy)).to_dict()
    }
    
    return df_copy, results


# ================== Feature engineering ==================

def encode_onehot_with_reference(df, column_name, prefix=None):
    """One-hot encoding with smallest category as reference (dropped)"""
    counts = df[column_name].value_counts()
    smallest_category = counts.index[-1]  # Get smallest category
    
    print(f"\n Encoding {column_name} with prefix {prefix}")
    print(f"Reference category (dropped): {smallest_category}")
    
    # Create one-hot encoded columns
    dummies = pd.get_dummies(df[column_name], prefix=prefix or column_name, drop_first=False)
    # Drop smallest category as reference
    dummies = dummies.drop(f'{prefix or column_name}_{smallest_category}', axis=1)

    # print the first 5 rows of the encoded variable
    print(dummies.head())

   
    # Concatenate with original dataframe
    df = pd.concat([df, dummies], axis=1)

    # print samples of the other category
    print(f"\nSamples of the other category:")
    mask = df[column_name] == smallest_category
    print(pd.concat([df[mask][[column_name]], dummies[mask]], axis=1).head())

    return df, dummies



# Frequency-based encoding for Country and Platform
def encode_frequency(df, column_name):
    freq = df[column_name].value_counts().to_dict()
    df[f'{column_name}_freq_encoded'] = df[column_name].map(freq)

    # print the first 5 rows of the encoding mapping   
    print(f"\n{column_name} frequency encoding (top 5):")
    for country, freq in list(freq.items())[:5]:
        print(f"{country}: {freq}")

    # print the first 5 rows of the encoded variable
    print(f"\n{column_name} frequency encoded variable (top 5):")
    print(df[[f'{column_name}_freq_encoded']].head())

    # plot encoded variable distribution
    plt.figure(figsize=(8, 3))
    sns.countplot(data=df, x=f'{column_name}_freq_encoded')
    plt.title(f'Distribution of {column_name} Encoded Variable')
    plt.ylabel('Count')
    plt.show()

    return df

# 1. Dictionary mapping each country to its continent
country_to_continent = {
    # Africa
    "Egypt": "Africa", "Morocco": "Africa", "South Africa": "Africa",
    "Nigeria": "Africa", "Kenya": "Africa", "Ghana": "Africa",
    # Asia
    "Bangladesh": "Asia", "India": "Asia", "China": "Asia",
    "Japan": "Asia", "South Korea": "Asia", "Malaysia": "Asia",
    "Thailand": "Asia", "Vietnam": "Asia", "Philippines": "Asia",
    "Indonesia": "Asia", "Taiwan": "Asia", "Hong Kong": "Asia",
    "Singapore": "Asia", "UAE": "Asia", "Israel": "Asia",
    "Turkey": "Asia", "Qatar": "Asia", "Kuwait": "Asia",
    "Bahrain": "Asia", "Oman": "Asia", "Jordan": "Asia",
    "Lebanon": "Asia", "Iraq": "Asia", "Yemen": "Asia",
    "Syria": "Asia", "Afghanistan": "Asia", "Pakistan": "Asia",
    "Nepal": "Asia", "Bhutan": "Asia", "Sri Lanka": "Asia",
    "Maldives": "Asia", "Kazakhstan": "Asia", "Uzbekistan": "Asia",
    "Kyrgyzstan": "Asia", "Tajikistan": "Asia", "Armenia": "Asia",
    "Georgia": "Asia", "Azerbaijan": "Asia", "Cyprus": "Asia",
    # Europe
    "UK": "Europe", "Germany": "Europe", "France": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Denmark": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Switzerland": "Europe", "Austria": "Europe",
    "Portugal": "Europe", "Greece": "Europe", "Ireland": "Europe",
    "Iceland": "Europe", "Finland": "Europe", "Poland": "Europe",
    "Romania": "Europe", "Hungary": "Europe", "Czech Republic": "Europe",
    "Slovakia": "Europe", "Croatia": "Europe", "Serbia": "Europe",
    "Slovenia": "Europe", "Bulgaria": "Europe", "Estonia": "Europe",
    "Latvia": "Europe", "Lithuania": "Europe", "Ukraine": "Europe",
    "Moldova": "Europe", "Belarus": "Europe", "Russia": "Europe",
    "Luxembourg": "Europe", "Monaco": "Europe", "Andorra": "Europe",
    "San Marino": "Europe", "Vatican City": "Europe",
    "Liechtenstein": "Europe", "Montenegro": "Europe", "Albania": "Europe",
    "North Macedonia": "Europe", "Kosovo": "Europe", "Bosnia": "Europe",
    # North America
    "USA": "North America", "Canada": "North America",
    "Mexico": "North America", "Costa Rica": "North America",
    "Panama": "North America", "Jamaica": "North America",
    "Trinidad": "North America", "Bahamas": "North America",
    # South America
    "Brazil": "South America", "Argentina": "South America",
    "Chile": "South America", "Colombia": "South America",
    "Peru": "South America", "Venezuela": "South America",
    "Ecuador": "South America", "Uruguay": "South America",
    "Paraguay": "South America", "Bolivia": "South America",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania"
}

# 2. Function to map a country to its continent
def cont_map(country):
    """
    Return the continent for a given country.
    If the country is not in the mapping, returns 'Other'.
    """
    return country_to_continent.get(country, "Other")

def map_to_continent(df, visualize=False):
    df["Continent"] = df["Country"].apply(cont_map)
    if visualize:
        plt.figure(figsize=(8, 3))
        sns.countplot(data=df, x='Continent')
        plt.title('Distribution of Continent Variable')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    return df



# Transformer to group rare categories into "Other"
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups categories whose absolute count in the training data is below `min_count`
    into a single 'Other' category. Accepts X as a 1-D array/Series or 2-D array/DataFrame.
    """
    def __init__(self, min_count=30):
        self.min_count = min_count
        self.frequent_categories_ = set()

    def fit(self, X, y=None):
        # Flatten to 1-D
        arr = np.array(X)
        flat = arr.ravel()
        series = pd.Series(flat).fillna("Missing")
        
        counts = series.value_counts()
        self.frequent_categories_ = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        arr = np.array(X)
        flat = arr.ravel()
        series = pd.Series(flat).fillna("Missing")
        out = series.where(series.isin(self.frequent_categories_), "Other")
        # Return as 2-D (n_samples, 1)
        return out.to_frame()
    

# Transformer to map country to continent
continent_dict = {
    # Africa
    "Egypt": "Africa", "Morocco": "Africa", "South Africa": "Africa",
    "Nigeria": "Africa", "Kenya": "Africa", "Ghana": "Africa",
    # Asia
    "Bangladesh": "Asia", "India": "Asia", "China": "Asia",
    "Japan": "Asia", "South Korea": "Asia", "Malaysia": "Asia",
    "Thailand": "Asia", "Vietnam": "Asia", "Philippines": "Asia",
    "Indonesia": "Asia", "Taiwan": "Asia", "Hong Kong": "Asia",
    "Singapore": "Asia", "UAE": "Asia", "Israel": "Asia",
    "Turkey": "Asia", "Qatar": "Asia", "Kuwait": "Asia",
    "Bahrain": "Asia", "Oman": "Asia", "Jordan": "Asia",
    "Lebanon": "Asia", "Iraq": "Asia", "Yemen": "Asia",
    "Syria": "Asia", "Afghanistan": "Asia", "Pakistan": "Asia",
    "Nepal": "Asia", "Bhutan": "Asia", "Sri Lanka": "Asia",
    "Maldives": "Asia", "Kazakhstan": "Asia", "Uzbekistan": "Asia",
    "Kyrgyzstan": "Asia", "Tajikistan": "Asia", "Armenia": "Asia",
    "Georgia": "Asia", "Azerbaijan": "Asia", "Cyprus": "Asia",
    # Europe
    "UK": "Europe", "Germany": "Europe", "France": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Denmark": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Switzerland": "Europe", "Austria": "Europe",
    "Portugal": "Europe", "Greece": "Europe", "Ireland": "Europe",
    "Iceland": "Europe", "Finland": "Europe", "Poland": "Europe",
    "Romania": "Europe", "Hungary": "Europe", "Czech Republic": "Europe",
    "Slovakia": "Europe", "Croatia": "Europe", "Serbia": "Europe",
    "Slovenia": "Europe", "Bulgaria": "Europe", "Estonia": "Europe",
    "Latvia": "Europe", "Lithuania": "Europe", "Ukraine": "Europe",
    "Moldova": "Europe", "Belarus": "Europe", "Russia": "Europe",
    "Luxembourg": "Europe", "Monaco": "Europe", "Andorra": "Europe",
    "San Marino": "Europe", "Vatican City": "Europe",
    "Liechtenstein": "Europe", "Montenegro": "Europe", "Albania": "Europe",
    "North Macedonia": "Europe", "Kosovo": "Europe", "Bosnia": "Europe",
    # North America
    "USA": "North America", "Canada": "North America",
    "Mexico": "North America", "Costa Rica": "North America",
    "Panama": "North America", "Jamaica": "North America",
    "Trinidad": "North America", "Bahamas": "North America",
    # South America
    "Brazil": "South America", "Argentina": "South America",
    "Chile": "South America", "Colombia": "South America",
    "Peru": "South America", "Venezuela": "South America",
    "Ecuador": "South America", "Uruguay": "South America",
    "Paraguay": "South America", "Bolivia": "South America",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania"
}
# class to map country to continent
class CountryToContinentMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X may be a 2D numpy array or DataFrame with shape (n_samples, 1)
        # Flatten it to a 1D array/Series first:
        vals = X.values if hasattr(X, 'values') else X
        flat = vals.ravel()               # shape (n_samples,)
        mapped = pd.Series(flat).map(self.mapping)
        return mapped.fillna("Other").to_frame()



# Helper to extract feature names even when some transformers lack get_feature_names_out
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, cols in column_transformer.transformers_:
        # Skip dropped columns or remainder
        if transformer == 'drop' or name == 'remainder':
            continue

        # Normalize cols into a list
        input_cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]

        # If it's a pipeline, grab its last step
        tr = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer

        # Attempt to get feature names
        if hasattr(tr, 'get_feature_names_out'):
            try:
                # First try passing the original column names
                names = tr.get_feature_names_out(input_cols)
            except Exception:
                try:
                    # Fallback to no-arg version
                    names = tr.get_feature_names_out()
                except Exception:
                    # Final fallback: use the input column names
                    names = input_cols
        else:
            # Transformer has no naming method
            names = input_cols

        feature_names.extend(names)
    return feature_names


# ===============================
# Classification Pipelines
# ===============================

def mlflow_dataset(X_train_full, X_test):
    train_ds = mlflow.data.from_pandas(
        df=X_train_full,
        source="../data/data_cleaned.pickle",
        name="social_sphere_train_v1"
    )
    test_ds = mlflow.data.from_pandas(
        df=X_test,
        source="../data/data_cleaned.pickle",
        name="social_sphere_test_v1"
    )
    return {"train_ds": train_ds, "test_ds": test_ds}

# The function to run a classification experiment
def run_classification_experiment(
    name: str,
    estimator,                # e.g. Pipeline([('preproc', preprocessor), ('clf', LogisticRegression(...))])
    X_train, y_train,
    cv,
    scoring,            # dict of scoring metrics
    dataset,
    hparams,
    registered_model_name: str = "conflict_baseline_dummy"
):
    with mlflow.start_run(run_name=name):
        # Log inputs
        mlflow.log_input(dataset["train_ds"], context="training")    
        mlflow.log_input(dataset["test_ds"], context="test")

        # log strategy
        mlflow.log_param("hyperparameters", hparams)

        # 2) Cross-validate & log CV metrics
        cv_results = cross_validate(
            estimator=estimator,
            X=X_train, y=y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                mlflow.log_metric(metric.replace("test_", ""), scores.mean().round(2))

        # 3) Re-fit on full train and register model
        estimator.fit(X_train, y_train)

        # 4) Infer signature & input example for better model packaging
        example_input = X_train.iloc[:5]
        preds = estimator.predict(example_input)
        signature = infer_signature(example_input, preds)

        mlflow.sklearn.log_model(
            sk_model=estimator,
            name=name,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=example_input
        )




# ===============================
# SHAP
# ===============================
def run_shap_experiment(
    best_model,
    X_train_full,
    feature_perturbation="interventional",
    plot_type="bar",
    shap_type="linear"
):
    # 1. Extract best model and its preprocessing step
    # best_model = grid_search_result.best_estimator_
    preprocessor = best_model.named_steps['preprocessing']
    classifier   = best_model.named_steps['classifier']

    # 2. Prepare data for SHAP
    #    Use a subset of training data (or validation) for faster computation
    X_shap_raw = X_train_full.sample(n=200, random_state=42)
    # Transform to model inputs
    X_shap_proc = preprocessor.transform(X_shap_raw)
    if sparse.issparse(X_shap_proc):
        X_shap_proc = X_shap_proc.toarray()

    # Recover feature names
    feature_names = get_feature_names(preprocessor)
    X_shap_df = pd.DataFrame(X_shap_proc, columns=feature_names)

    if shap_type == "linear":    
        # Initialize a SHAP explainer
        explainer = shap.LinearExplainer(
            model=classifier,
            masker=X_shap_df,
            feature_perturbation=feature_perturbation   # supported: "interventional" or "correlation_dependent"
        )
    else:
        explainer = shap.TreeExplainer(
            model=classifier,              # the XGBClassifier instance
            data=X_shap_df,                # background data for expected values
            feature_perturbation=feature_perturbation  # optional
        )

    # Compute SHAP values
    shap_values = explainer.shap_values(X_shap_df)

    # Summary plot for the positive class (if binary)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_shap_df, plot_type=plot_type, show=False
    )
    plt.title("SHAP Summary Plot")
    # plt.show()
    fig_sum = plt.gcf()
    plt.close(fig_sum)
    return fig_sum

# ===============================
# log_test_set_performance
# ===============================

def log_test_set_performance(
    model,
    X_test,
    y_test,
    prefix: str = "test"
):
    """
    Evaluate `model` on (X_test, y_test), compute common metrics,
    a confusion matrix plot, and a text classification report,
    then log everything into the active MLflow run.
    
    Handles both binary and multiclass classification automatically.
    """
    # 1) Predictions & probabilities
    y_pred = model.predict(X_test)
    
    # Determine if this is binary or multiclass
    n_classes = len(np.unique(y_test))
    is_binary = n_classes == 2
    
    # 2) Compute metrics based on problem type
    if is_binary:
        # Binary classification metrics
        proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        metrics = {
            f"{prefix}_accuracy":   accuracy_score(y_test, y_pred),
            f"{prefix}_precision":  precision_score(y_test, y_pred),
            f"{prefix}_recall":     recall_score(y_test, y_pred),
            f"{prefix}_f1_score":   f1_score(y_test, y_pred),
            f"{prefix}_roc_auc":    roc_auc_score(y_test, proba),
        }
    else:
        # Multiclass classification metrics
        metrics = {
            f"{prefix}_accuracy":        accuracy_score(y_test, y_pred),
            f"{prefix}_precision_macro": precision_score(y_test, y_pred, average='macro'),
            f"{prefix}_precision_weighted": precision_score(y_test, y_pred, average='weighted'),
            f"{prefix}_recall_macro":    recall_score(y_test, y_pred, average='macro'),
            f"{prefix}_recall_weighted": recall_score(y_test, y_pred, average='weighted'),
            f"{prefix}_f1_macro":        f1_score(y_test, y_pred, average='macro'),
            f"{prefix}_f1_weighted":     f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Add multiclass ROC-AUC if model supports predict_proba
        try:
            proba = model.predict_proba(X_test)
            metrics[f"{prefix}_roc_auc_ovr"] = roc_auc_score(y_test, proba, multi_class='ovr')
            metrics[f"{prefix}_roc_auc_ovo"] = roc_auc_score(y_test, proba, multi_class='ovo')
        except:
            # Some models might not support predict_proba
            pass
    
    # 3) Log metrics
    for name, val in metrics.items():
        mlflow.log_metric(name, round(val, 4))

    # 4) Classification report
    report = classification_report(y_test, y_pred)
    report_path = f"{prefix}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # 5) Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create labels for confusion matrix
    if is_binary:
        labels = ['Low', 'High']
    else:
        # For 3-class: 0=Low, 1=Medium, 2=High
        labels = ['Low', 'Medium', 'High']
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{prefix.capitalize()} Confusion Matrix")
    plt.tight_layout()
    
    cm_path = f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifact(cm_path)
    
    # Clean up temporary files
    import os
    if os.path.exists(report_path):
        os.remove(report_path)
    if os.path.exists(cm_path):
        os.remove(cm_path)


# ===============================
# Grid Search
# ===============================

def run_classification_gridsearch_experiment(
    name: str,
    pipeline,               # e.g. Pipeline([... , ('classifier', LogisticRegression())])
    param_grid: dict,
    X_train, y_train, X_test, y_test,
    cv,                     # e.g. StratifiedKFold(...)
    scoring,                # e.g. {"accuracy":"accuracy", "f1_score":"f1", ...}
    dataset: dict,          # {"train_ds": X_train_full, "test_ds": X_test}
    registered_model_name: str,
    verbose: bool = False,
    feature_set: str = "all",
    refit_metric: str = "f1_score"
):
    grid_search = GridSearchCV(
        estimator=pipeline,        # your Pipeline([... ('classifier', LogisticRegression()) ])
        param_grid=param_grid,
        cv=cv,                         # 5-fold stratified CV
        scoring=scoring,            # primary metric
        n_jobs=-1,                    # parallelize across all cores
        return_train_score=True,
        refit=refit_metric,            # ← what to optimize/return as best_estimator_
        error_score=np.nan,  # treat fold‐errors as NaN rather than crashing
        verbose=1
    )

    print(f"Running grid search for {name}")
    with mlflow.start_run(run_name=name):
        # 1) Log dataset inputs
        mlflow.log_input(dataset["train_ds"], context="training")
        mlflow.log_input(dataset["test_ds"],  context="test")

        # 2) Log the grid search hyperparameter space
        mlflow.log_param("grid_search_params", param_grid)
        
        # Log the feature set
        mlflow.log_param("feature_set", feature_set)

        grid_search.fit(X_train, y_train)

        # 4) Log the best hyperparameters
        best_params = grid_search.best_params_
        mlflow.log_param("hyperparameters", best_params)

        # 5) Log CV metrics (mean ± std) for each scoring key
        results = grid_search.cv_results_
        # if scoring is a dict, iterate its keys; else assume single metric
        for metric in scoring:
            mean_score = results[f"mean_test_{metric}"][grid_search.best_index_]
            std_score  = results[f"std_test_{metric}"][grid_search.best_index_]
            mlflow.log_metric(metric, mean_score.round(2))
            mlflow.log_metric(f"{metric}_std", std_score)
            if verbose:
                print(f"{metric}: {mean_score.round(2)} ± {std_score.round(2)}")

        # 6) Log and register the best estimator
        best_estimator = grid_search.best_estimator_
        example_input = X_train.iloc[:5]
        example_preds = best_estimator.predict(example_input)
        signature = infer_signature(example_input, example_preds)
        
        # ===== SHAP plot for binary classification
        n_classes = len(np.unique(y_train))
        is_binary = n_classes == 2
        if is_binary:
            plot_type = "violin"
        else:
            plot_type = "bar"
        # Adding shap to the model
        # print(f"Running SHAP for {name}")
        fig_sum = None
        if 'baseline' not in name:
            if 'logreg' in name:
                shap_type = "linear"
            else:
                shap_type = "tree"
            # print(f"Running SHAP for {name} with {plot_type} and {shap_type}")
            fig_sum = run_shap_experiment(
                best_model=best_estimator,
                X_train_full=X_train,
                feature_perturbation="interventional",
                plot_type=plot_type, # other options: "bar", "dot", "violin"
                shap_type=shap_type
            )
            mlflow.log_figure(fig_sum, "shap_summary.png")

        # ===== ROC plot for binary classification
        fig_roc = None
        if is_binary:
            fig_roc = create_and_log_roc_plot_binary(
                model=best_estimator,
                X_test=X_test,
                y_test=y_test,
                prefix="test",
                show_plot=False
            )
            mlflow.log_figure(fig_roc, "roc_curve.png")


        # Log test set performance
        log_test_set_performance(
            model=best_estimator,
            X_test=X_test,
            y_test=y_test,
            prefix="test"
        )

        # 7) Log the best estimator
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            name=name,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=example_input
        )
    if verbose:
        # 3. Print summary in the output cell
        print("Best parameters:", grid_search.best_params_)
        print("Best F1 score (CV):", grid_search.best_score_.round(2))

        # 4. (Optional) inspect all CV results
        cv_df = pd.DataFrame(grid_search.cv_results_)
        display(cv_df.sort_values("mean_test_f1_score", ascending=False).head(3))
    
    # plot SHAP summary
    # if 'baseline' not in name and fig_sum is not None:
    #     plt.figure(figsize=(10, 6))
    #     plt.title("SHAP Summary Plot")
    #     plt.show()

    return best_estimator, grid_search, fig_sum, fig_roc


# ===============================
# Multi-class classification
# ===============================

def create_multiclass_conflict(df, target_column='Conflicts', low_threshold=1, high_threshold=3, visualize=True):
    """
    Create 3-class conflict classification (Low, Medium, High) from conflict scores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, default='Conflicts'
        Name of the conflict column
    low_threshold : int, default=1
        Upper bound for Low class (inclusive). Low: 0 to low_threshold
    high_threshold : int, default=3  
        Lower bound for High class (exclusive). High: > high_threshold
        Medium: low_threshold+1 to high_threshold
    visualize : bool, default=True
        Whether to create visualizations
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added 3-class conflict column
    dict
        Dictionary with analysis results including thresholds, counts, and class balance
    """
    # Make a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Create 3-class target variable
    def classify_conflict(score):
        if score <= low_threshold:
            return 'Low'
        elif score <= high_threshold:
            return 'Medium' 
        else:
            return 'High'
    
    df_copy['Conflict_3Class'] = df_copy[target_column].apply(classify_conflict)
    
    # Calculate statistics
    conflict_counts = df_copy['Conflict_3Class'].value_counts()
    total_samples = len(df_copy)
    
    # Print analysis
    print(f"3-Class Conflict Classification:")
    print(f"Low Conflict (0-{low_threshold}): {conflict_counts.get('Low', 0)} samples")
    print(f"Medium Conflict ({low_threshold+1}-{high_threshold}): {conflict_counts.get('Medium', 0)} samples")
    print(f"High Conflict ({high_threshold+1}-max): {conflict_counts.get('High', 0)} samples")
    print(f"Class proportions:")
    proportions = conflict_counts / total_samples
    for class_name, prop in proportions.items():
        print(f"  {class_name}: {prop:.3f} ({prop*100:.1f}%)")
    
    # Convert to numeric labels (0, 1, 2)
    class_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df_copy['Conflict_3Class_Numeric'] = df_copy['Conflict_3Class'].map(class_mapping)
    
    # Create visualizations if requested
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original distribution with threshold lines
        axes[0].hist(df_copy[target_column], bins=range(int(df_copy[target_column].min()), 
                                                       int(df_copy[target_column].max()) + 2), 
                    alpha=0.7, edgecolor='black')
        axes[0].axvline(low_threshold + 0.5, color='red', linestyle='--', linewidth=2,
                       label=f'Low/Medium = {low_threshold + 0.5}')
        axes[0].axvline(high_threshold + 0.5, color='orange', linestyle='--', linewidth=2,
                       label=f'Medium/High = {high_threshold + 0.5}')
        axes[0].set_xlabel('Conflict Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Original Conflict Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 3-class distribution (categorical)
        sns.countplot(data=df_copy, x='Conflict_3Class', ax=axes[1], 
                     order=['Low', 'Medium', 'High'])
        axes[1].set_title('3-Class Conflict Distribution\n(Categorical Labels)')
        axes[1].set_ylabel('Count')
        
        # Add count labels on bars
        for i, bar in enumerate(axes[1].patches):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 3-class distribution (numeric)
        sns.countplot(data=df_copy, x='Conflict_3Class_Numeric', ax=axes[2])
        axes[2].set_title('3-Class Conflict Distribution\n(0=Low, 1=Medium, 2=High)')
        axes[2].set_ylabel('Count')
        axes[2].set_xlabel('Class (Numeric)')
        
        # Add count labels on bars
        for i, bar in enumerate(axes[2].patches):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    # Prepare results dictionary
    results = {
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'counts': conflict_counts.to_dict(),
        'proportions': proportions.to_dict(),
        'class_mapping': class_mapping
    }
    
    return df_copy, results

# Converting targets from binary to multi-class 
def convert_to_multiclass_target(y_series, original_data, target_column='Conflicts', 
                                low_threshold=1, high_threshold=3):
    """
    Convert existing target variables to 3-class format without recreating train/test splits.
    
    Parameters:
    -----------
    y_series : pd.Series or array-like
        The target variable series (can be binary or original conflicts)
    original_data : pd.DataFrame  
        Original dataframe with the Conflicts column
    target_column : str, default='Conflicts'
        Name of the original conflict column
    low_threshold : int, default=1
        Upper bound for Low class (inclusive)
    high_threshold : int, default=3
        Lower bound for High class (exclusive)
        
    Returns:
    --------
    pd.Series
        3-class target variable (0=Low, 1=Medium, 2=High)
    """
    # Get the original conflict scores for the same indices as y_series
    original_conflicts = original_data.loc[y_series.index, target_column]
    
    # Create 3-class labels
    def classify_conflict(score):
        if score <= low_threshold:
            return 0  # Low
        elif score <= high_threshold:
            return 1  # Medium
        else:
            return 2  # High
    
    multiclass_target = original_conflicts.apply(classify_conflict)
    
    # Print distribution
    counts = multiclass_target.value_counts().sort_index()
    print(f"3-Class Distribution:")
    print(f"Low (0): {counts.get(0, 0)} samples")
    print(f"Medium (1): {counts.get(1, 0)} samples") 
    print(f"High (2): {counts.get(2, 0)} samples")
    
    return multiclass_target


def create_and_log_roc_plot_binary(model, X_test, y_test, prefix="test", show_plot=True):
    """
    Create ROC plot for binary classification and log to MLflow.
    """
    from sklearn.metrics import roc_curve, auc
    
    # Get predicted probabilities for positive class
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    fig_roc = plt.gcf()
    
    # Log to MLflow
    roc_path = f"{prefix}_roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    mlflow.log_artifact(roc_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig_roc)
    
    # Clean up
    # import os
    if os.path.exists(roc_path):
        os.remove(roc_path)
    
        # Show in notebook if requested
    # if show_plot:
        # plt.show()
    # else:
        # plt.close(fig)
    
    return fig_roc