# Social Sphere LLM Project

A comprehensive LLM (Large Language Model) project using Jupyter notebooks for social media analysis and natural language processing tasks.

## 🚀 Features

- **Data Analysis**: Pandas and NumPy for data manipulation and analysis
- **Machine Learning**: Scikit-learn for traditional ML algorithms
- **Deep Learning**: PyTorch and Transformers for LLM development
- **Visualization**: Matplotlib and Seaborn for data visualization
- **Experiment Tracking**: Weights & Biases integration
- **Development Tools**: Jupyter notebooks for interactive development

## 📋 Prerequisites

- Python 3.8 or higher
- `uv` package manager (recommended) or `pip`

## 🛠️ Installation

1. **Clone the repository** (if using git):
   ```bash
   git clone <repository-url>
   cd Social-Sphere2
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate
   
   # Or using pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Using uv
   uv pip install -e .
   
   # Or using pip
   pip install -e .
   ```

4. **Install development dependencies** (optional):
   ```bash
   # Using uv
   uv pip install -e ".[dev]"
   
   # Or using pip
   pip install -e ".[dev]"
   ```

## 📦 Project Structure

```
Social-Sphere2/
├── .venv/                 # Virtual environment
├── notebooks/             # Jupyter notebooks
├── data/                  # Data files
├── models/                # Trained models
├── src/                   # Source code
├── tests/                 # Test files
├── pyproject.toml         # Project configuration
├── README.md             # This file
└── .gitignore            # Git ignore file
```

## 🎯 Usage

1. **Start Jupyter Lab**:
   ```bash
   jupyter lab
   ```

2. **Or start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Create your first notebook**:
   - Navigate to the `notebooks/` directory
   - Create a new Python notebook
   - Start experimenting with the installed packages

## 📚 Key Dependencies

### Core Data Science
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms

### Deep Learning & LLM
- **torch**: PyTorch deep learning framework
- **transformers**: Hugging Face transformers library
- **datasets**: Hugging Face datasets library
- **accelerate**: Distributed training utilities
- **tokenizers**: Fast tokenization library

### Visualization
- **matplotlib**: Basic plotting library
- **seaborn**: Statistical data visualization

### Development & Monitoring
- **jupyter**: Interactive notebooks
- **wandb**: Experiment tracking
- **tensorboard**: Training visualization
- **tqdm**: Progress bars

## 🔧 Development

### Code Formatting
```bash
# Format code with black
black .

# Sort imports with isort
isort .
```

### Running Tests
```bash
pytest
```

### Linting
```bash
flake8 .
```

## 📊 Example Usage

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch

# Load and preprocess data
df = pd.read_csv('data/your_data.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Load pre-trained model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Your LLM training code here...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions, please:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include your environment details and error messages

## 🔄 Updates

Keep your dependencies up to date:
```bash
uv pip install --upgrade -e .
```

---

**Happy coding! 🎉** 