import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Pie chart for conflict prediction

def create_conflict_pie_chart(result):
    fig, ax = plt.subplots(figsize=(3, 2))
    
    # Handle missing confidence
    if 'confidence' in result and result['confidence'] is not None:
        confidence = result['confidence']
    else:
        confidence = 0.8  # Default confidence when not available
    
    if result['conflict_level'] == 'High Risk':
        colors = ['#ff6b6b', '#4ecdc4']
        sizes = [confidence, 1 - confidence]
        labels = ['High Risk', 'Low Risk']
    else:
        colors = ['#4ecdc4', '#ff6b6b']
        sizes = [confidence, 1 - confidence]
        labels = ['Low Risk', 'High Risk']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, explode=(0.1, 0))
    
    # Update title to handle missing confidence
    if 'confidence' in result and result['confidence'] is not None:
        title = f'Conflict Risk Prediction\nConfidence: {confidence:.1%}'
    else:
        title = f'Conflict Risk Prediction\nConfidence: {confidence:.1%} (estimated)'
    
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.legend(wedges, labels, title="Risk Levels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{img_base64}"

# Histogram for addiction score (not currently used, but included for completeness)
def create_addiction_score_chart(result, data=None):
    fig, ax = plt.subplots(figsize=(3, 2))
    if data is not None and 'Addicted_Score' in data.columns:
        scores = data['Addicted_Score'].dropna()
    else:
        np.random.seed(42)
        scores = np.random.normal(5.5, 1.5, 1000)
        scores = np.clip(scores, 1, 10)
    n, bins, patches = ax.hist(scores, bins=20, alpha=0.7, color='#4ecdc4',
                              edgecolor='black', linewidth=0.5)
    predicted_score = result['predicted_score']
    ax.axvline(x=predicted_score, color='#ff6b6b', linewidth=3,
               label=f'Your Prediction: {predicted_score:.2f}')
    if 'confidence' in result:
        confidence = result['confidence']
        ax.axvspan(predicted_score - 0.5, predicted_score + 0.5,
                  alpha=0.3, color='#ff6b6b',
                  label=f'Confidence: {confidence:.2f}')
    ax.set_title('Addiction Score Distribution with Your Prediction',
                fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel('Addiction Score (1-10)', fontsize=8, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=8, fontweight='bold')
    ax.axvspan(1, 3, alpha=0.2, color='green', label='Low Addiction (1-3)')
    ax.axvspan(3, 7, alpha=0.2, color='orange', label='Moderate Addiction (3-7)')
    ax.axvspan(7, 10, alpha=0.2, color='red', label='High Addiction (7-10)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{img_base64}"

# Gauge chart for addiction score
def create_addiction_gauge_chart(result):
    fig, ax = plt.subplots(figsize=(3, 2), subplot_kw={'projection': 'polar'})
    predicted_score = result['predicted_score']
    angle = (predicted_score - 1) * 20
    theta = np.linspace(0, np.pi, 100)
    # Reduce black line thickness by 30% (from 3 to 2.1)
    ax.plot(theta, [1]*100, 'k-', linewidth=2.1)
    # Flip colors over y-axis - now high scores are on the left (0) and low scores on the right (pi)
    high_angle = np.linspace(0, 2*20*np.pi/180, 50)
    ax.fill_between(high_angle, 0, 1, alpha=0.3, color='red', label='High (7-10)')
    mod_angle = np.linspace(2*20*np.pi/180, 6*20*np.pi/180, 50)
    ax.fill_between(mod_angle, 0, 1, alpha=0.3, color='orange', label='Moderate (3-7)')
    low_angle = np.linspace(6*20*np.pi/180, np.pi, 50)
    ax.fill_between(low_angle, 0, 1, alpha=0.3, color='green', label='Low (1-3)')
    needle_angle = angle * np.pi / 180
    # Reduce needle line thickness by 30% (from 4 to 2.8)
    ax.plot([needle_angle, needle_angle], [0, 1.2], 'k-', linewidth=2.8, label=f'Your Score: {predicted_score:.1f}')
    # Reduce round dot size by 50% (from 10 to 5)
    ax.plot(needle_angle, 1.2, 'ko', markersize=5, markeredgecolor='white', markeredgewidth=2)
    ax.set_title(f'Addiction Score Gauge\nPredicted: {predicted_score:.1f}/10',
                fontsize=8, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.3)
    # Flip the gauge by adjusting text positions and angles
    ax.text(np.pi, 1.4, 'Low\n(1-3)', ha='center', va='center', fontsize=3, fontweight='bold')
    ax.text(np.pi/2, 1.4, 'Moderate\n(3-7)', ha='center', va='center', fontsize=3, fontweight='bold')
    ax.text(0, 1.4, 'High\n(7-10)', ha='center', va='center', fontsize=3, fontweight='bold')
    if 'confidence' in result:
        confidence = result['confidence']
        ax.text(0, -0.3, f'Confidence: {confidence:.2f}', ha='center', va='center',
               fontsize=6, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{img_base64}"

# Clustering: Elbow and Sleep vs Age scatter plot
def create_clustering_charts(result, cluster_df=None, user_sleep=None, user_age=None, user_cluster=None, cluster_labels_map=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    optimal_k = 3
    k_values = range(1, 11)
    inertias = [150, 120, 85, 65, 55, 50, 47, 45, 43, 42]
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax1.set_ylabel('Inertia', fontweight='bold')
    ax1.set_title('Elbow Method: Optimal K Selection', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k = {optimal_k}')
    ax1.legend()
    # Scatter plot: Sleep vs Age colored by cluster (always 3 clusters/colors)
    if cluster_df is not None and cluster_labels_map is not None:
        colors = ['#4ecdc4', '#ffd93d', '#ff6b6b']
        for cluster in range(optimal_k):
            sub = cluster_df[cluster_df['cluster'] == cluster]
            color = colors[cluster % len(colors)]
            label = cluster_labels_map.get(cluster, f'Cluster {cluster}')
            ax2.scatter(sub['Sleep_Hours_Per_Night'], sub['Age'], c=color, alpha=0.7, s=50, label=label)
        # Highlight the user's point
        if user_sleep is not None and user_age is not None and user_cluster is not None:
            color = colors[user_cluster % len(colors)]
            ax2.scatter([user_sleep], [user_age], c='red', marker='*', s=250, edgecolors='black', linewidths=2, label='You')
    ax2.set_xlabel('Sleep Hours per Night', fontweight='bold')
    ax2.set_ylabel('Age', fontweight='bold')
    ax2.set_title(f'Cluster Analysis: Sleep vs Age (k={optimal_k})', fontsize=12, fontweight='bold')
    handles, labels = ax2.get_legend_handles_labels()
    # Remove duplicate 'You' label if present
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax2.legend(new_handles, new_labels)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{img_base64}"