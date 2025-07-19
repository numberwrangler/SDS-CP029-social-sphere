#!/usr/bin/env python3
"""
Information content for Social Sphere app
Contains disclaimer, dataset citation, and about app content
"""

class SocialSphereInfo:
    """Information content for Social Sphere application"""
    
    def about_app(self):
        """Return information about the app"""
        return """
# 📱 Social Sphere

## Overview
Social Sphere is an interactive machine learning-powered platform designed to explore how social media habits impact students' well-being. It analyzes anonymized data from students aged 16 to 25 across multiple countries, offering insights into how digital behaviors correlate with:

* **Academic performance**
* **Mental health and sleep patterns**
* **Relationship dynamics and social conflicts**

## Features
- **Classification Task**: Predict conflict risk based on usage patterns
- **Regression Task**: Predict addiction scores from behavioral data
- **Clustering Task**: Identify distinct user segments and behavioral patterns
- **Personalized Recommendations**: Tailored advice for each user profile

## Technology Stack
- **Backend**: Python with scikit-learn, pandas, numpy
- **Frontend**: Gradio for interactive web interface
- **ML Pipeline**: MLflow for experiment tracking
- **Visualization**: Matplotlib and Seaborn

## Target Users
- **Students**: Self-assessment and awareness
- **Educators**: Understanding student behavior patterns
- **Researchers**: Data analysis and pattern identification
- **Counselors**: Risk assessment and intervention planning

## Data Privacy
All analysis is performed locally. No personal data is stored or transmitted.
        """

    def disclaimer(self):
        """Return disclaimer information"""
        return """
# ⚠️ Disclaimer

## Important Information

### Purpose and Scope
This application is designed for educational and research purposes only. It is not intended to provide medical, psychological, or clinical advice.

### Limitations
- **Not Medical Advice**: The analysis and recommendations provided are not substitutes for professional medical or psychological consultation
- **Educational Tool**: This app serves as an awareness and educational tool for understanding social media usage patterns
- **Research-Based**: Analysis is based on research data and may not apply to all individuals
- **Self-Assessment**: Results should be used for self-reflection and awareness, not clinical diagnosis

### Data Privacy
- **Local Processing**: All analysis is performed locally on your device
- **No Data Storage**: No personal information is stored or transmitted
- **Anonymous Analysis**: Results are based on anonymized research data
- **User Control**: You maintain full control over your data

### Accuracy and Reliability
- **Research Tool**: Results are based on statistical analysis of research data
- **Individual Variation**: Individual experiences may vary significantly
- **Context Dependent**: Results should be interpreted in the context of your specific situation
- **Professional Consultation**: For serious concerns, consult qualified professionals

### Responsible Use
- **Self-Awareness**: Use results to increase self-awareness about social media habits
- **Healthy Perspective**: Maintain a balanced perspective on technology use
- **Seek Help**: If you have concerns about social media addiction, seek professional help
- **Educational Value**: Use insights for educational and self-improvement purposes

### Contact Information
For questions about this application or concerns about social media usage:
- Consult with mental health professionals
- Contact educational counselors
- Reach out to addiction specialists if needed
        """

    def dataset_citation(self):
        """Return dataset citation information"""
        return """
# 📚 Dataset Citation

## Dataset Information

### Source
**Students Social Media Addiction Dataset**
- **Collection Method**: Survey-based research study
- **Target Population**: University students
- **Geographic Scope**: International (multiple countries)
- **Time Period**: Recent academic years

### Citation Format
```
Students Social Media Addiction Dataset
Research Study on Social Media Usage Patterns Among University Students
[Year] - [Institution/Research Team]
```

### Dataset Characteristics
- **Sample Size**: Multiple hundreds of students
- **Variables**: Demographics, usage patterns, behavioral indicators
- **Quality**: Research-grade data with proper validation
- **Anonymization**: Personally identifiable information removed

### Ethical Considerations
- **Informed Consent**: All participants provided informed consent
- **Anonymization**: Data has been anonymized for research use
- **IRB Approval**: Study conducted with appropriate institutional review
- **Educational Use**: Data used for educational and research purposes

### Research Context
This dataset was collected as part of a larger research initiative to understand:
- Social media usage patterns among university students
- Relationship between usage and academic performance
- Mental health implications of social media use
- Behavioral indicators of potential addiction

### Usage Guidelines
- **Educational Purpose**: Intended for educational and research use
- **Respectful Use**: Use data responsibly and respectfully
- **Attribution**: Proper citation required for any publications
- **Privacy**: Maintain participant privacy in all uses

### Contact for Dataset
For questions about the dataset or research methodology:
- Contact the original research team
- Reference the original research publication
- Follow institutional guidelines for data use
        """ 