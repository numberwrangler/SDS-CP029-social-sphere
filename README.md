# Welcome to the SuperDataScience Community Project!
Welcome to the **Social Sphere: Student Social-Media Behavior & Relationship Analytics** repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Project Scope of Works:
1.
2.
3.

## Project Overview
**Social Sphere** explores how social-media habits relate to students’ academic performance, sleep, mental health, and relationship dynamics. Using an anonymized cross-section of 16- to 25-year-olds from multiple countries, the project will:

- surface global insights on usage intensity, platform preference, and well-being;
- predict relationship conflicts and self-reported addiction levels;
- segment students into behavior-based clusters;
- deliver an interactive Streamlit app for researchers, educators, and mental-health advocates.

**Link to Dataset:** https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships

## Project Objectives
### Exploratory Data Analysis
- Profile demographics (age, gender, academic level, country) and social-media metrics.
- Visualize links between **Avg_Daily_Usage_Hours**, **Mental_Health_Score**, and **Sleep_Hours_Per_Night**.
- Assess country-level and platform-level differences.
- Handle missing values, encode categoricals, and flag potential biases.

### Model Development
1. **Predictive Tasks**
    - **Conflict Classifier**: predict `Conflicts_Over_Social_Media` (high vs. low) from usage patterns, addiction score, and platform.
    - **Addiction-Level Regressor**: estimate `Addicted_Score` as a continuous outcome.

2. **Clustering**
    - Group students into interpretable segments (e.g., “high-usage high-stress”) with K-Means / HDBSCAN on scaled features.

3. **Evaluation**
    - Classification: Accuracy, Precision, Recall, F1, ROC-AUC.
    - Regression: MAE, RMSE, R².
    - Clustering: Silhouette Score, qualitative segment validation.

4. **Hyperparameter Tuning**
    - Grid / random search to optimize tree-based and gradient-boosted models.


### Model Deployment
- **Streamlit Web App**
    - Dashboards (usage heatmaps, correlations, country comparisons).
    - Live predictor for conflict likelihood and addiction score.
    - Cluster explorer with segment descriptions.

- Host on **Streamlit Community Cloud** (or similar) with public README and user guide.

## Technical Requirements
### Tools and Libraries
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `mlflow`
- **Clustering & Dimensionality Reduction**: `KMeans`, `HDBSCAN`, `PCA/UMAP`
- **Deployment**: `Streamlit`
### Environment
- Python 3.9+
- Virtual environment (`conda` or `venv`)


## Workflow

1. **Setup & EDA (Week 1)**
    - Set up GitHub repository, project structure, and virtual environment.
    - Data profiling, outlier checks, visualization, and bias assessment.

2. **Feature Engineering & Model Development (Weeks 2 - 4)**
    - Encode categoricals, scale and transform numerical features, derive interaction features.
    - Train and tune classifiers and regressors.
    - Build clustering pipeline.
    - Experiment tracking using ML Flow

3. **Deployment (Week 5)**
    - Build Streamlit UI (Insights, Predictors, Clusters) and deploy to the cloud.

## Timeline

| Phase                                           | Core Tasks                                                                                               | Duration        |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------- |
| **1 · Setup & EDA**                             | GitHub repo, project structure, virtual environment, data profiling, outlier checks, visualization       | **Week 1**      |
| **2 · Feature Engineering & Model Development** | Design, train, experiment with models.                                                                   | **Weeks 2 - 4** |
| **3 · Deployment**                              | Build Streamlit UI (Insights, Predictors, Clusters); deploy to cloud                                     | **Week 5**      |

