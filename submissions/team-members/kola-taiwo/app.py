# Importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_models, validate_input, preprocess_input, test_models

# App Config
st.set_page_config(
    page_title="Social Media Impact Investigation",
    page_icon="��",
    layout="wide"
)

# Adding a banner image with title overlay
st.markdown("""
<div style="position: relative; text-align: center; margin-bottom: 30px;">
    <img src="https://images.unsplash.com/photo-1611224923853-80b023f02d71?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1200&q=80" 
         style="width: 100%; height: 300px; object-fit: cover; border-radius: 10px;">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px;">
        <h1 style="color: white; font-size: 2.5rem; margin: 0; font-weight: bold;">Social Sphere Students Behaviour</h1>
        <p style="color: white; font-size: 1.1rem; margin: 10px 0 0 0; opacity: 0.9;">Analyzing Social Media Impact on Student Life</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Title and Description
st.title("Social Media Impact Investigation")
st.markdown("""
This app analyzes the impact of social media usage on Academic Performance, Mental Health, 
and Relationships using machine learning models.
""")

# Loading Data and Models
@st.cache_resource
def load_data_and_models():
    try:
        # Loading data
        df = pd.read_csv('data/ssma.csv')
        
        # Loading models
        models = load_models(
            regressor_path='models/regressor_model.pkl',
            classifier_path='models/classifier_model.pkl',
            cluster_path='models/clustering_model.pkl'
        )
        
        # Testing models
        test_models(*models)
        
        return df, models
    except Exception as e:
        st.error(f"Error loading data or models: {str(e)}")
        st.stop()

df, models = load_data_and_models()
regressor, classifier, cluster_model, preprocessor = models

# Sidebar - User Input
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 15, 30, 20)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    academic_level = st.sidebar.selectbox('Academic Level', 
                                        ['High School', 'Undergraduate', 'Graduate'])
    country = st.sidebar.selectbox('Country', sorted(df['Country'].unique()))
    usage_hours = st.sidebar.slider('Average Daily Usage (hours)', 1.0, 15.0, 5.0)
    platform = st.sidebar.selectbox('Most Used Platform', 
                                   ['Instagram', 'Facebook', 'TikTok', 'Twitter', 'YouTube'])
    affects_academic = st.sidebar.radio('Affects Academic Performance?', ['Yes', 'No'])
    sleep_hours = st.sidebar.slider('Sleep Hours Per Night', 3.0, 12.0, 7.0)
    mental_health = st.sidebar.slider('Mental Health Score (1-10)', 1, 10, 6)
    relationship_status = st.sidebar.selectbox('Relationship Status', 
                                             ['Single', 'In Relationship', 'Complicated'])
    
    data = {
        'Age': age,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Country': country,
        'Avg_Daily_Usage_Hours': usage_hours,
        'Most_Used_Platform': platform,
        'Affects_Academic_Performance': affects_academic,
        'Sleep_Hours_Per_Night': sleep_hours,
        'Mental_Health_Score': mental_health,
        'Relationship_Status': relationship_status
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main Panel
tab1, tab2, tab3 = st.tabs(["EDA Insights", "Model Predictions", "About"])

with tab1:
    st.header("Exploratory Data Analysis")
    
    # Correlation Heatmap
    st.subheader("Feature Correlation")
    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    
    # Country Comparison
    st.subheader("Country Comparison")
    country_col, metric_col = st.columns(2)
    
    with country_col:
        countries = st.multiselect(
            "Select countries to compare",
            df['Country'].unique(),
            default=df['Country'].unique()[:3]
        )
    
    with metric_col:
        metric = st.selectbox(
            "Select metric to compare",
            ['Avg_Daily_Usage_Hours', 'Mental_Health_Score', 'Sleep_Hours_Per_Night']
        )
    
    if countries:
        filtered = df[df['Country'].isin(countries)]
        fig = px.box(filtered, x='Country', y=metric, color='Country')
        st.plotly_chart(fig, use_container_width=True)
    
    # Usage Distribution
    st.subheader("Daily Usage Distribution")
    fig = px.histogram(df, x='Avg_Daily_Usage_Hours', nbins=20, 
                      title='Distribution of Daily Social Media Usage')
    st.plotly_chart(fig, use_container_width=True)
    
    # Platform Usage
    st.subheader("Most Used Platforms")
    platform_counts = df['Most_Used_Platform'].value_counts()
    fig = px.pie(values=platform_counts.values, names=platform_counts.index, 
                title='Distribution of Most Used Platforms')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Predictions")
    
    if st.button('Run Analysis'):
        try:
            # Validate input
            validate_input(input_df)
            
            # Preprocess input
            processed_input = preprocess_input(input_df, preprocessor)
            
            # Make predictions
            addiction_score = regressor.predict(processed_input)[0]
            conflict_level = classifier.predict(processed_input)[0]
            cluster = cluster_model.predict(processed_input)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Addiction Score", f"{addiction_score:.1f}/10")
                st.progress(addiction_score/10)
                st.caption("Higher scores indicate higher risk of addiction")
            
            with col2:
                conflict_label = "High" if conflict_level == 1 else "Low"
                st.metric("Conflict Level", conflict_label)
                st.caption("Likelihood of relationship conflicts due to social media")
            
            with col3:
                cluster_info = {
                    0: {"name": "Balanced Users", "intervention": "Maintain current habits"},
                    1: {"name": "At-Risk Users", "intervention": "Reduce usage by 2 hours/day"},
                    2: {"name": "High-Risk Users", "intervention": "Seek professional guidance"}
                }
                st.metric("Cluster Group", cluster_info[cluster]['name'])
                st.info(f"Recommendation: {cluster_info[cluster]['intervention']}")
            
            # Feature importance
            st.subheader("Key Influencing Factors")
            
            if hasattr(regressor, 'feature_importances_'):
                # Get feature names from preprocessor
                feature_names = []
                for name, trans, cols in preprocessor.transformers_:
                    if hasattr(trans, 'get_feature_names_out'):
                        feature_names.extend(trans.get_feature_names_out(cols))
                    else:
                        feature_names.extend(cols)
                
                importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': regressor.feature_importances_
                }).sort_values('Importance', ascending=False).head(5)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

with tab3:
    st.header("About This App")
    st.markdown("""
    ### User Guide
    
    1. **Input Features**: Fill in your details in the sidebar
    2. **EDA Insights**: Explore correlations and patterns in the data
    3. **Model Predictions**: Click "Run Analysis" to get predictions
    
    ### Models Used
    - **Addiction Score Predictor**: Random Forest Regressor
    - **Conflict Level Classifier**: Random Forest Classifier
    - **User Clustering**: K-Means (3 clusters)
    
    ### Features Analyzed
    - Age, Gender, Academic Level
    - Country, Daily Usage Hours
    - Most Used Platform
    - Academic Performance Impact
    - Sleep Hours, Mental Health Score
    - Relationship Status
    
    ### Deployment
    To run this app locally:
    ```
    cd submissions/team-members/kola-taiwo
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """) 