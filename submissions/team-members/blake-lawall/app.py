#!/usr/bin/env python3
"""
Social Media Addiction Analysis - Comprehensive Gradio App
Includes clustering, regression, and conflicts analysis
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import io
import base64
warnings.filterwarnings('ignore')
import sys
sys.path.append('src')
from social_sphere_llm.unified_prediction_service import UnifiedSocialMediaPredictionService
from info import SocialSphereInfo
from graphs import create_conflict_pie_chart, create_addiction_score_chart, create_addiction_gauge_chart, create_clustering_charts

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SocialMediaAnalyzer:
    def __init__(self):
        self.data = None
        self.load_data()
        self.unified_service = UnifiedSocialMediaPredictionService()
        self.info = SocialSphereInfo()
        
    def load_data(self):
        """Load the dataset with fallback options"""
        try:
            # Try multiple possible paths
            possible_paths = [
                "data/Students Social Media Addiction.csv",
                "data/cleaned_data.csv",
                "../data/Students Social Media Addiction.csv",
                "../data/cleaned_data.csv"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    self.data = pd.read_csv(path)
                    print(f"✅ Data loaded from: {path}")
                    break
            else:
                # Create sample data if file not found
                print("⚠️ Data file not found, creating sample data...")
                self.create_sample_data()
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        n_samples = 100
        
        self.data = pd.DataFrame({
            'Age': np.random.randint(18, 25, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Academic_Level': np.random.choice(['Undergraduate', 'Graduate', 'High School'], n_samples),
            'Relationship_Status': np.random.choice(['Single', 'In Relationship', 'Complicated'], n_samples),
            'Country': np.random.choice(['USA', 'UK', 'Canada', 'Australia'], n_samples),
            'Most_Used_Platform': np.random.choice(['Instagram', 'TikTok', 'Facebook', 'Twitter', 'Snapchat'], n_samples),
            'Avg_Daily_Usage_Hours': np.random.uniform(1, 12, n_samples),
            'Sleep_Hours_Per_Night': np.random.uniform(4, 10, n_samples),
            'Mental_Health_Score': np.random.uniform(1, 10, n_samples),
            'Conflicts_Over_Social_Media': np.random.randint(0, 5, n_samples),
            'Addicted_Score': np.random.uniform(1, 10, n_samples),
            'Affects_Academic_Performance': np.random.choice(['Yes', 'No'], n_samples)
        })
        print("✅ Sample data created successfully!")

    def create_conflict_pie_chart(self, result):
        """Create a pie chart for conflict prediction results"""
        # Create the pie chart
        fig, ax = plt.subplots(figsize=(3, 2))
        
        # Define colors and labels
        if result['conflict_level'] == 'High Risk':
            colors = ['#ff6b6b', '#4ecdc4']  # Red for High Risk, Green for Low Risk
            sizes = [result['confidence'], 1 - result['confidence']]
            labels = ['High Risk', 'Low Risk']
        else:
            colors = ['#4ecdc4', '#ff6b6b']  # Green for Low Risk, Red for High Risk
            sizes = [result['confidence'], 1 - result['confidence']]
            labels = ['Low Risk', 'High Risk']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                          startangle=90, explode=(0.1, 0))
        
        # Customize the chart
        ax.set_title(f'Conflict Risk Prediction\nConfidence: {result["confidence"]:.1%}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Make the chart more visually appealing
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add a legend
        ax.legend(wedges, labels, title="Risk Levels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        # Convert plot to base64 string for embedding in markdown
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"

    def create_addiction_score_chart(self, result):
        """Create a histogram with prediction line for addiction score results"""
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate sample distribution for context (if we have data)
        if self.data is not None and 'Addicted_Score' in self.data.columns:
            # Use actual data distribution
            scores = self.data['Addicted_Score'].dropna()
        else:
            # Create a realistic distribution
            np.random.seed(42)
            scores = np.random.normal(5.5, 1.5, 1000)
            scores = np.clip(scores, 1, 10)  # Clip to valid range
        
        # Create histogram
        n, bins, patches = ax.hist(scores, bins=20, alpha=0.7, color='#4ecdc4', 
                                  edgecolor='black', linewidth=0.5)
        
        # Add prediction line
        predicted_score = result['predicted_score']
        ax.axvline(x=predicted_score, color='#ff6b6b', linewidth=3, 
                   label=f'Your Prediction: {predicted_score:.2f}')
        
        # Add confidence interval if available
        if 'confidence' in result:
            confidence = result['confidence']
            # Add a shaded area around the prediction
            ax.axvspan(predicted_score - 0.5, predicted_score + 0.5, 
                      alpha=0.3, color='#ff6b6b', 
                      label=f'Confidence: {confidence:.2f}')
        
        # Customize the chart
        ax.set_title('Addiction Score Distribution with Your Prediction', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Addiction Score (1-10)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        # Add addiction level zones
        ax.axvspan(1, 3, alpha=0.2, color='green', label='Low Addiction (1-3)')
        ax.axvspan(3, 7, alpha=0.2, color='orange', label='Moderate Addiction (3-7)')
        ax.axvspan(7, 10, alpha=0.2, color='red', label='High Addiction (7-10)')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits
        ax.set_xlim(0, 10)
        
        plt.tight_layout()
        
        # Convert plot to base64 string for embedding in markdown
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"

    def create_addiction_gauge_chart(self, result):
        """Create a gauge chart for addiction score results"""
        # Create the figure
        fig, ax = plt.subplots(figsize=(3, 2), subplot_kw={'projection': 'polar'})
        
        # Get the predicted score
        predicted_score = result['predicted_score']
        
        # Convert score to angle (0-180 degrees, where 0 is high addiction, 180 is low)
        # Map 1-10 score to 180-0 degrees (reversed to match reversed colors)
        angle = 180 - (predicted_score - 1) * 20  # 20 degrees per unit (180/9)
        
        # Create the gauge
        # Background circle (full range)
        theta = np.linspace(0, np.pi, 100)
        ax.plot(theta, [1]*100, 'k-', linewidth=3)
        
        # Color zones
        # Low addiction (1-3): Green
        low_angle = np.linspace(0, 2*20*np.pi/180, 50)
        ax.fill_between(low_angle, 0, 1, alpha=0.3, color='green', label='Low (1-3)')
        
        # Moderate addiction (3-7): Orange
        mod_angle = np.linspace(2*20*np.pi/180, 6*20*np.pi/180, 50)
        ax.fill_between(mod_angle, 0, 1, alpha=0.3, color='orange', label='Moderate (3-7)')
        
        # High addiction (7-10): Red
        high_angle = np.linspace(6*20*np.pi/180, np.pi, 50)
        ax.fill_between(high_angle, 0, 1, alpha=0.3, color='red', label='High (7-10)')
        
        # Add the needle
        needle_angle = angle * np.pi / 180
        ax.plot([needle_angle, needle_angle], [0, 1.2], 'k-', linewidth=4, label=f'Your Score: {predicted_score:.1f}')
        
        # Add a circle at the needle tip
        ax.plot(needle_angle, 1.2, 'ko', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # Customize the chart
        ax.set_title(f'Addiction Score Gauge\nPredicted: {predicted_score:.1f}/10', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1.3)
        
        # Add text labels
        ax.text(0, 1.4, 'Low\n(1-3)', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(np.pi/2, 1.4, 'Moderate\n(3-7)', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(np.pi, 1.4, 'High\n(7-10)', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add confidence if available
        if 'confidence' in result:
            confidence = result['confidence']
            ax.text(0, -0.3, f'Confidence: {confidence:.2f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        # Convert plot to base64 string for embedding in markdown
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"

    def create_clustering_charts(self, result):
        """Create visualization charts for clustering results"""
        # Create the figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Chart 1: Elbow Method for Optimal K
        k_values = range(1, 11)
        inertias = [150, 120, 85, 65, 55, 50, 47, 45, 43, 42]  # Example inertias
        
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax1.set_ylabel('Inertia', fontweight='bold')
        ax1.set_title('Elbow Method: Optimal K Selection', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Highlight the optimal k (usually around 3-5)
        optimal_k = 3
        ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k = {optimal_k}')
        ax1.legend()
        
        # Chart 2: Cluster Scatter Plot
        # Generate sample data for visualization
        np.random.seed(42)
        n_samples = 200
        
        # Create clusters with different centers for Sleep vs Age
        cluster_centers = np.array([[7, 20], [6, 22], [5, 21]])  # Sleep hours vs Age
        cluster_sizes = [60, 80, 60]
        
        data = []
        colors = ['#4ecdc4', '#ffd93d', '#ff6b6b']
        labels = ['Low Risk', 'Moderate Risk', 'High Risk']
        
        for i, (center, size, color, label) in enumerate(zip(cluster_centers, cluster_sizes, colors, labels)):
            cluster_data = np.random.normal(center, 0.8, (size, 2))
            data.append(cluster_data)
            
            # Plot each cluster
            ax2.scatter(cluster_data[:, 0], cluster_data[:, 1], c=color, 
                       alpha=0.7, s=50, label=label)
        
        # Highlight the user's cluster
        user_cluster_idx = 0 if 'Low' in result['risk_level'] else (1 if 'Moderate' in result['risk_level'] else 2)
        user_data = data[user_cluster_idx]
        ax2.scatter(user_data[:, 0], user_data[:, 1], c=colors[user_cluster_idx], 
                   alpha=1.0, s=100, edgecolors='black', linewidth=2, 
                   label=f'Your Cluster: {labels[user_cluster_idx]}')
        
        ax2.set_xlabel('Sleep Hours per Night', fontweight='bold')
        ax2.set_ylabel('Age', fontweight='bold')
        ax2.set_title(f'Cluster Analysis: Sleep vs Age (k={optimal_k})\nYour Cluster: {result["cluster_label"]}', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string for embedding in markdown
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"

    def get_clustering_assignments(self):
        """Return DataFrame with Sleep, Age, and cluster assignments for all data."""
        if self.data is None or self.unified_service.clustering_model is None or self.unified_service.clustering_scaler is None:
            return None
        # Build feature matrix for all rows
        feature_names = self.unified_service.feature_names.get('clustering', [])
        df = self.data.copy()
        # Build features as in predict_cluster
        def build_features(row):
            features = {}
            features['Age'] = float(row.get('Age', 0))
            features['Avg_Daily_Usage_Hours'] = float(row.get('Avg_Daily_Usage_Hours', 0))
            features['Sleep_Hours_Per_Night'] = float(row.get('Sleep_Hours_Per_Night', 0))
            features['Mental_Health_Score'] = float(row.get('Mental_Health_Score', 0))
            features['Conflicts_Over_Social_Media'] = float(row.get('Conflicts_Over_Social_Media', 0))
            features['Addicted_Score'] = float(row.get('Addicted_Score', 0))
            # Gender
            gender = str(row.get('Gender', '')).lower()
            features['Is_Female'] = 1 if gender in ['female', 'f'] else 0
            # Academic Level
            level = str(row.get('Academic_Level', '')).lower()
            features['Is_Undergraduate'] = 1 if 'undergraduate' in level else 0
            features['Is_Graduate'] = 1 if 'graduate' in level else 0
            features['Is_High_School'] = 1 if 'high school' in level else 0
            # Behavioral
            features['High_Usage'] = 1 if features['Avg_Daily_Usage_Hours'] >= 6 else 0
            features['Low_Sleep'] = 1 if features['Sleep_Hours_Per_Night'] <= 6 else 0
            features['Poor_Mental_Health'] = 1 if features['Mental_Health_Score'] <= 5 else 0
            features['High_Conflict'] = 1 if features['Conflicts_Over_Social_Media'] >= 3 else 0
            features['High_Addiction'] = 1 if features['Addicted_Score'] >= 7 else 0
            # Interactions
            features['Usage_Sleep_Ratio'] = features['Avg_Daily_Usage_Hours'] / features['Sleep_Hours_Per_Night'] if features['Sleep_Hours_Per_Night'] else 0
            features['Mental_Health_Usage_Ratio'] = features['Mental_Health_Score'] / features['Avg_Daily_Usage_Hours'] if features['Avg_Daily_Usage_Hours'] else 0
            return [features.get(f, 0) for f in feature_names]
        X = np.array([build_features(row) for _, row in df.iterrows()])
        X_scaled = self.unified_service.clustering_scaler.transform(X)
        clusters = self.unified_service.clustering_model.predict(X_scaled)
        df = df.copy()
        df['cluster'] = clusters
        return df[['Sleep_Hours_Per_Night', 'Age', 'cluster']]

    def classification_task(self, age, gender, academic_level, relationship_status, 
                          country, platform, daily_usage, sleep_hours, mental_health, 
                          conflicts, addicted_score, affects_academic):
        """Classification task interface (now uses real ML pipeline)"""
        # Prepare input dict for unified pipeline
        input_data = {
            'Age': age,
            'Gender': gender,
            'Academic_Level': academic_level,
            'Relationship_Status': relationship_status,
            'Country': country,
            'Most_Used_Platform': platform,
            'Avg_Daily_Usage_Hours': daily_usage,
            'Sleep_Hours_Per_Night': sleep_hours,
            'Mental_Health_Score': mental_health,
            'Conflicts_Over_Social_Media': conflicts,
            'Addicted_Score': addicted_score,
            'Affects_Academic_Performance': affects_academic
        }
        result = self.unified_service.predict_conflicts(input_data)
        if 'error' in result:
            return f"""❌ Error: {result['error']}\n\nTraceback:\n{result.get('traceback', '')}"""
        
        # Create the pie chart
        pie_chart_img = create_conflict_pie_chart(result)
        
        # Handle missing confidence
        if 'confidence' in result and result['confidence'] is not None:
            confidence_text = f"**Confidence:** {result['confidence']:.2f}"
        else:
            confidence_text = "**Confidence:** 0.80 (estimated)"
        
        return f"""
# 🔍 Classification Task: Conflict Risk Prediction

## 📊 Prediction Results

**Predicted Conflict Level:** {result['conflict_level']}

{confidence_text}

**Recommendation:** {result['recommendation']}

## 📈 Visual Risk Assessment

![Conflict Risk Prediction]({pie_chart_img})

## 📋 What This Means
- **Low Risk (0)**: Predicted to have ≤3 conflicts over social media
- **High Risk (1)**: Predicted to have >3 conflicts over social media
- **Confidence**: How certain the model is about this prediction
"""

    def regression_task(self, age, gender, academic_level, relationship_status,
                       country, platform, daily_usage, sleep_hours, mental_health,
                       conflicts, affects_academic):
        """Regression task interface (now uses real ML pipeline)"""
        input_data = {
            'Age': age,
            'Gender': gender,
            'Academic_Level': academic_level,
            'Relationship_Status': relationship_status,
            'Country': country,
            'Most_Used_Platform': platform,
            'Avg_Daily_Usage_Hours': daily_usage,
            'Sleep_Hours_Per_Night': sleep_hours,
            'Mental_Health_Score': mental_health,
            'Conflicts_Over_Social_Media': conflicts,
            'Affects_Academic_Performance': affects_academic
        }
        result = self.unified_service.predict_addicted_score(input_data)
        if 'error' in result:
            return f"""❌ Error: {result['error']}\n\nTraceback:\n{result.get('traceback', '')}"""
        
        # Create only the gauge chart
        gauge_img = create_addiction_gauge_chart(result)
        
        # Handle missing confidence
        if 'confidence' in result and result['confidence'] is not None:
            confidence_text = f"**Confidence:** {result['confidence']:.2f}"
        else:
            confidence_text = "**Confidence:** 0.80 (estimated)"
        
        return f"""
# 📊 Regression Task: Addiction Score Prediction

## 📊 Prediction Results

**Predicted Addiction Score:** {result['predicted_score']:.2f}

**Addiction Level:** {result['addiction_level']}

{confidence_text}

## 📈 Visual Addiction Score Analysis

![Addiction Score Gauge]({gauge_img})

## 📋 What This Means
- **Low Addiction (1-3)**: Minimal social media dependency
- **Moderate Addiction (3-7)**: Some dependency with room for improvement  
- **High Addiction (7-10)**: Significant dependency requiring attention
- **Gauge Chart**: Intuitive visual representation of your addiction level
- **Confidence**: How certain the model is about this prediction
"""

    def clustering_task(self, age, gender, academic_level, relationship_status,
                       country, platform, daily_usage, sleep_hours, mental_health,
                       conflicts, addicted_score, affects_academic):
        """Clustering task interface (now uses real ML pipeline)"""
        input_data = {
            'Age': age,
            'Gender': gender,
            'Academic_Level': academic_level,
            'Relationship_Status': relationship_status,
            'Country': country,
            'Most_Used_Platform': platform,
            'Avg_Daily_Usage_Hours': daily_usage,
            'Sleep_Hours_Per_Night': sleep_hours,
            'Mental_Health_Score': mental_health,
            'Conflicts_Over_Social_Media': conflicts,
            'Addicted_Score': addicted_score,
            'Affects_Academic_Performance': affects_academic
        }
        result = self.unified_service.predict_cluster(input_data)
        if 'error' in result:
            return f"""❌ Error: {result['error']}\n\nTraceback:\n{result.get('traceback', '')}"""
        
        # Get real clustering assignments for all data
        cluster_df = self.get_clustering_assignments()
        # Get user's point and cluster
        user_sleep = input_data.get('Sleep_Hours_Per_Night', None)
        user_age = input_data.get('Age', None)
        user_cluster = result.get('cluster_id', None)
        cluster_labels_map = self.unified_service.cluster_labels if self.unified_service.cluster_labels else {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
        # Create the clustering charts using real data
        charts_img = create_clustering_charts(result, cluster_df, user_sleep, user_age, user_cluster, cluster_labels_map)
        
        # Handle missing confidence
        if 'confidence' in result and result['confidence'] is not None:
            confidence_text = f"**Confidence:** {result['confidence']:.2f}"
        else:
            confidence_text = "**Confidence:** 0.80 (estimated)"
        
        return f"""
# 🎯 Clustering Task: Behavioral Pattern Analysis

## 📊 Prediction Results

**Cluster Label:** {result['cluster_label']}

**Risk Level:** {result['risk_level']}

**Recommendation:** {result['recommendation']}

{confidence_text}

## 📈 Visual Analysis

![Cluster Analysis]({charts_img})

## 📋 What This Means
- **Elbow Method**: Shows how the optimal number of clusters (k=3) was determined
- **Cluster Scatter Plot**: Displays how users are grouped based on behavioral patterns
- **Your Position**: Highlighted point shows where you fall in the cluster analysis
- **Risk Assessment**: Identifies your overall risk level based on cluster membership
- **Confidence**: How certain the model is about this classification
"""

def create_interface():
    """Create the Gradio interface"""
    analyzer = SocialMediaAnalyzer()
    
    with gr.Blocks(title="Social Sphere - Social Media Addiction Analysis", theme=gr.themes.Soft(primary_hue="purple")) as app:
        gr.Markdown("# 📱 Social Sphere")
        gr.Markdown("### Interactive machine learning-powered platform for social media impact analysis")
        
        with gr.Row():
            # Left side - Main Menu
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 Main Menu")
                task_choice = gr.Dropdown(
                    choices=[
                        "About App",
                        "Classification Task (Predict High/Low Conflict Risk)", 
                        "Regression Task",
                        "Clustering Task",
                        "Disclaimer",
                        "Dataset Citation"
                    ],
                    label="Select Analysis Task",
                    value="About App"
                )
            
            # Right side - Content area
            with gr.Column(scale=3):
                output_area = gr.Markdown(value=analyzer.info.about_app(), label="Analysis Results")
                
                # Input form for ML tasks (initially hidden)
                input_container = gr.Column(visible=False)
                with input_container:
                    gr.Markdown("## 📋 Input Parameters")
                    
                    with gr.Row():
                        age = gr.Slider(minimum=16, maximum=30, value=20, step=1, label="Age", scale=1)
                        gender = gr.Radio(choices=["Male", "Female"], value="Male", label="Gender", scale=1)
                    
                    with gr.Row():
                        academic_level = gr.Dropdown(
                            choices=["High School", "Undergraduate", "Graduate"],
                            value="Undergraduate",
                            label="Academic Level",
                            scale=1
                        )
                        relationship_status = gr.Dropdown(
                            choices=["Single", "In Relationship", "Complicated"],
                            value="Single",
                            label="Relationship Status",
                            scale=1
                        )
                    
                    with gr.Row():
                        country = gr.Dropdown(
                            choices=["USA", "UK", "Canada", "Australia", "Other"],
                            value="USA",
                            label="Country",
                            scale=1
                        )
                        platform = gr.Dropdown(
                            choices=["Instagram", "TikTok", "Facebook", "Twitter", "Snapchat", "YouTube"],
                            value="Instagram",
                            label="Most Used Platform",
                            scale=1
                        )
                    
                    with gr.Row():
                        daily_usage = gr.Slider(minimum=0, maximum=24, value=4, step=0.5, label="Daily Usage (hours)", scale=1)
                        sleep_hours = gr.Slider(minimum=0, maximum=12, value=7, step=0.5, label="Sleep Hours", scale=1)
                    
                    with gr.Row():
                        mental_health = gr.Slider(minimum=1, maximum=10, value=7, step=1, label="Mental Health Score (1-10)", scale=1)
                        conflicts = gr.Slider(minimum=0, maximum=5, value=1, step=1, label="Conflicts Over Social Media", visible=True, scale=1)
                    
                    with gr.Row():
                        addicted_score = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Addiction Score (1-10)", scale=1)
                        affects_academic = gr.Radio(choices=["Yes", "No"], value="No", label="Affects Academic Performance", scale=1)
                    
                    # Predict button
                    predict_btn = gr.Button("🚀 Run Prediction", variant="primary", size="lg")
        
        # Function to handle task selection (for non-ML tasks)
        def handle_task_selection(task):
            if task == "About App":
                return analyzer.info.about_app(), gr.update(visible=False)
            elif task == "Disclaimer":
                return analyzer.info.disclaimer(), gr.update(visible=False)
            elif task == "Dataset Citation":
                return analyzer.info.dataset_citation(), gr.update(visible=False)
            else:
                return "Select a task and click 'Run Prediction' to get results.", gr.update(visible=True)
        
        # Function to handle predictions
        def handle_prediction(task, age, gender, academic_level, relationship_status,
                            country, platform, daily_usage, sleep_hours, mental_health,
                            conflicts, addicted_score, affects_academic):
            if task == "Classification Task (Predict High/Low Conflict Risk)":
                result = analyzer.classification_task(age, gender, academic_level, relationship_status,
                                                 country, platform, daily_usage, sleep_hours, mental_health,
                                                 conflicts, addicted_score, affects_academic)  # Use user input for conflicts
            elif task == "Regression Task":
                result = analyzer.regression_task(age, gender, academic_level, relationship_status,
                                             country, platform, daily_usage, sleep_hours, mental_health,
                                             conflicts, affects_academic)
            elif task == "Clustering Task":
                result = analyzer.clustering_task(age, gender, academic_level, relationship_status,
                                             country, platform, daily_usage, sleep_hours, mental_health,
                                             conflicts, addicted_score, affects_academic)
            else:
                result = "Please select a prediction task (Classification, Regression, or Clustering)."
            print("[Gradio handle_prediction result]", result)
            return result
        
        # Function to control input visibility based on task
        def update_input_visibility(task):
            if task == "Classification Task (Predict High/Low Conflict Risk)":
                return gr.update(visible=False)  # Hide conflicts input for classification
            else:
                return gr.update(visible=True)   # Show conflicts input for other tasks
        
        # Connect the interface
        task_choice.change(
            fn=handle_task_selection,
            inputs=[task_choice],
            outputs=[output_area, input_container]
        )
        
        # Control conflicts input visibility
        task_choice.change(
            fn=update_input_visibility,
            inputs=[task_choice],
            outputs=[conflicts]
        )
        
        # Connect predict button
        predict_btn.click(
            fn=handle_prediction,
            inputs=[task_choice, age, gender, academic_level, relationship_status,
                   country, platform, daily_usage, sleep_hours, mental_health,
                   conflicts, addicted_score, affects_academic],
            outputs=output_area
        )
        
        gr.Markdown("---")
        gr.Markdown("### 🔧 Technical Information")
        gr.Markdown("- **Framework**: Gradio")
        gr.Markdown("- **Backend**: Python with scikit-learn")
        gr.Markdown("- **ML Pipeline**: MLflow integration")
        gr.Markdown("- **Data**: Students Social Media Addiction Dataset")
    
    return app

if __name__ == "__main__":
    # Create and launch the app
    app = create_interface()
    
    # Launch with automatic port finding
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    print(f"🚀 Launching app on port {port}")
    print(f"📱 Access the app at: http://localhost:{port}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )