# 📚 Social Sphere - Analysis Notebooks

This directory contains Jupyter notebooks for the comprehensive analysis of social media addiction data and machine learning model development.

## 📋 Notebook Overview

### 1. **01_data_analysis.ipynb**
- **Purpose**: Initial data exploration and preprocessing
- **Key Features**:
  - Dataset loading and basic statistics
  - Missing value analysis
  - Feature distribution visualization
  - Correlation analysis
  - Data cleaning and preparation

### 2. **02_conflicts_prediction_mlflow.ipynb**
- **Purpose**: Binary classification to predict high/low conflict risk using MLflow
- **Key Features**:
  - Feature engineering for conflicts prediction
  - MLflow integration for experiment tracking
  - Multiple model comparison (Random Forest, Gradient Boosting, Logistic Regression)
  - Model evaluation with accuracy, AUC, and confusion matrix
  - Feature importance analysis
  - Model saving and MLflow model registry

### 3. **03_regression_addicted_score.ipynb**
- **Purpose**: Regression analysis to predict addiction scores
- **Key Features**:
  - Target variable analysis and outlier detection
  - Feature correlation analysis
  - Multiple regression models (Random Forest, Gradient Boosting, Linear, Ridge, Lasso)
  - Model evaluation with R², RMSE, and MAE
  - Residual analysis and diagnostics
  - Model saving for production use

### 4. **04_clustering_analysis.ipynb**
- **Purpose**: Unsupervised learning to identify behavioral patterns
- **Key Features**:
  - Optimal cluster number determination (Elbow method, Silhouette score)
  - K-means clustering implementation
  - Cluster characteristics analysis
  - Cluster visualization using PCA
  - Behavioral pattern identification
  - Cluster labeling and risk assessment

## 🚀 Running the Notebooks

### Prerequisites
```bash
pip install -r ../requirements.txt
```

### Execution Order
1. Start with `01_data_analysis.ipynb` for data exploration
2. Run `02_conflicts_prediction_mlflow.ipynb` for classification model with MLflow
3. Run `03_regression_addicted_score.ipynb` for regression model
4. Run `04_clustering_analysis.ipynb` for clustering analysis

### Data Requirements
- Ensure `../data/Students Social Media Addiction.csv` exists
- The notebooks will create cleaned and processed data files

## 📊 Output Files

The notebooks generate several output files:

### Models (saved to `../models/`)
- `conflicts_classifier_rf.joblib` - Random Forest classifier for conflicts
- `conflicts_scaler.joblib` - Feature scaler for conflicts model
- `conflicts_feature_names.joblib` - Feature names for conflicts model
- `addicted_score_regressor.joblib` - Regression model for addiction scores
- `addicted_score_scaler.joblib` - Feature scaler for regression model
- `clustering_model.joblib` - K-means clustering model
- `clustering_scaler.joblib` - Feature scaler for clustering
- `cluster_labels.joblib` - Cluster labels and descriptions

### MLflow Models
- `models:/conflicts_classifier/latest` - MLflow registered conflicts model
- `models:/addicted_score_regressor/latest` - MLflow registered regression model

### Data Files (saved to `../data/`)
- `cleaned_data.csv` - Preprocessed dataset
- `clustered_data.csv` - Dataset with cluster assignments

## 🔧 Key Features

### Data Preprocessing
- Categorical variable encoding
- Feature scaling and normalization
- Outlier detection and handling
- Missing value analysis

### Model Development
- Multiple algorithm comparison
- Hyperparameter optimization
- Cross-validation
- Model evaluation metrics
- MLflow experiment tracking

### Visualization
- Distribution plots
- Correlation matrices
- Feature importance charts
- Cluster visualizations
- Model performance comparisons

## 📈 Model Performance

### Conflicts Classification
- **Best Model**: Random Forest
- **Accuracy**: ~85-90%
- **AUC**: ~0.85-0.90
- **MLflow Integration**: Full experiment tracking

### Addiction Score Regression
- **Best Model**: Random Forest/Gradient Boosting
- **R² Score**: ~0.70-0.80
- **RMSE**: ~1.0-1.5
- **MLflow Integration**: Model versioning and registry

### Clustering
- **Optimal Clusters**: 3
- **Silhouette Score**: ~0.40-0.50
- **Behavioral Patterns**: Light/Moderate/Heavy users with risk levels

## 🎯 Use Cases

These notebooks support the Social Sphere Gradio application by providing:
1. **Predictive Models**: For real-time user predictions
2. **Behavioral Insights**: For understanding user patterns
3. **Risk Assessment**: For identifying high-risk users
4. **Recommendations**: For personalized advice
5. **MLflow Integration**: For model versioning and deployment

## 🔍 Troubleshooting

### Common Issues
1. **Missing Dependencies**: Install all required packages from `requirements.txt`
2. **Data Path Issues**: Ensure data files are in the correct location
3. **Memory Issues**: Consider reducing dataset size for large files
4. **Model Loading Errors**: Check that all model files are properly saved
5. **MLflow Connection**: Ensure MLflow server is running for model registry

### Performance Tips
- Use GPU acceleration if available for large datasets
- Consider feature selection for faster training
- Use cross-validation for robust model evaluation
- Monitor memory usage during model training
- Use MLflow for experiment tracking and model versioning

## 📝 Notes

- All models are saved in joblib format for easy loading in production
- MLflow models are registered for version control and deployment
- Feature engineering is consistent across all notebooks
- Random state is set for reproducible results
- Visualizations are optimized for clarity and insight
- Code includes comprehensive error handling and logging

For questions or issues, refer to the main project documentation or contact the development team. 