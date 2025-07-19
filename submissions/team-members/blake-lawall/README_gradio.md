# Social Media Addiction Analysis - Gradio App

A comprehensive web application for analyzing student social media usage patterns and providing personalized recommendations.

## Features

### 🔍 Individual Analysis
- **Personalized Assessment**: Enter your social media usage data for customized analysis
- **Cluster Assignment**: See which behavioral group you belong to
- **Risk Factor Identification**: Identify potential addiction and mental health concerns
- **Personalized Recommendations**: Receive actionable advice for healthier social media use

### 📊 Dashboard
- **Usage Distribution**: Visualize daily social media usage patterns
- **Mental Health Correlation**: Explore relationships between usage and mental health
- **Cluster Analysis**: See student distribution across behavioral clusters
- **Platform Usage**: Analyze most popular social media platforms
- **Cluster Characteristics**: Heatmap of cluster characteristics

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Ensure your dataset is in the correct location:
```
data/
└── Students Social Media Addiction.csv
```

### 3. Run the App
```bash
python app.py
```

The app will be available at:
- **Local**: http://localhost:7860
- **Public**: A shareable link will be provided

## Usage

### Individual Analysis
1. Go to the "🔍 Individual Analysis" tab
2. Fill in your social media usage information:
   - Age, Gender, Academic Level
   - Daily usage hours, Sleep hours
   - Mental health score, Addiction score
   - Platform preferences, Conflicts
3. Click "🔍 Analyze My Usage"
4. Review your personalized results and recommendations

### Dashboard
1. Go to the "📊 Dashboard" tab
2. Click "📊 Generate Dashboard"
3. Explore the interactive visualizations:
   - Usage distribution patterns
   - Mental health correlations
   - Cluster characteristics
   - Platform preferences

## Key Metrics

- **Daily Usage**: Hours spent on social media per day
- **Mental Health Score**: Self-reported mental health (1-10 scale)
- **Sleep Hours**: Average sleep duration per night
- **Addiction Score**: Self-reported addiction level (1-10 scale)
- **Conflicts**: Number of conflicts related to social media use

## Risk Assessment

The app identifies risk factors based on:
- High daily usage (≥6 hours)
- Low sleep (≤6 hours)
- Poor mental health (≤5/10)
- High conflicts (≥3)
- High addiction score (≥7/10)

## Recommendations

The app provides personalized recommendations including:
- Setting daily usage limits
- Improving sleep hygiene
- Seeking mental health support
- Developing healthy digital boundaries
- Digital detox strategies

## Deployment Options

### Local Development
```bash
python app.py
```

### Gradio Spaces (Recommended)
1. Create a new Space on Hugging Face
2. Upload your files:
   - `app.py`
   - `requirements.txt`
   - `data/Students Social Media Addiction.csv`
3. Set the Space to use Gradio SDK
4. Deploy automatically

### Heroku
1. Create a `Procfile`:
```
web: python app.py
```
2. Deploy using Heroku CLI or GitHub integration

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

## Technical Details

### Architecture
- **Frontend**: Gradio (Python-based web framework)
- **Backend**: Python with scikit-learn for ML
- **Visualization**: Plotly for interactive charts
- **Clustering**: K-Means algorithm for user segmentation

### Data Processing
- Feature engineering for categorical variables
- Standardization for clustering
- Risk factor calculation
- Personalized recommendation generation

### Models
- **K-Means Clustering**: 4 clusters for user segmentation
- **StandardScaler**: Feature normalization
- **Risk Assessment**: Multi-factor evaluation

## Customization

### Adding New Features
1. Modify the `SocialMediaAnalyzer` class
2. Add new visualization functions
3. Update the Gradio interface

### Changing Clusters
1. Modify the `n_clusters` parameter in `train_models()`
2. Update cluster interpretation logic
3. Adjust risk assessment thresholds

### Adding New Platforms
1. Update platform choices in the interface
2. Add platform-specific features
3. Modify clustering features

## Troubleshooting

### Common Issues
1. **Data not found**: Ensure CSV file is in `data/` directory
2. **Import errors**: Install all requirements with `pip install -r requirements.txt`
3. **Port conflicts**: Change port in `app.py` launch parameters
4. **Memory issues**: Reduce dataset size or optimize data loading

### Performance Tips
- Use sample data for testing
- Optimize data loading for large datasets
- Cache model predictions
- Use async processing for heavy computations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes.

## Contact

For questions or support, please refer to the project documentation or create an issue in the repository. 