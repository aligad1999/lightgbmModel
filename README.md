# Risk Prediction Model with LightGBM

A comprehensive risk prediction model implementation using LightGBM with advanced techniques for handling imbalanced data.

## Features

- **SMOTEENN** for handling imbalanced datasets
- **PCA** for dimensionality reduction (applied to numerical features only)
- **LightGBM** gradient boosting with native categorical feature support
- **TimeSeriesSplit** for temporal cross-validation
- **Optuna** for Bayesian hyperparameter optimization

## Project Structure

```
.
├── risk_prediction_lightgbm.ipynb  # Main notebook with complete implementation
├── requirements.txt                 # Python dependencies
├── risk_prediction_model.pkl        # Saved trained model (generated after running)
├── model_metadata.json             # Model configuration and performance metrics
└── README.md                       # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd risk-prediction-model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Open and run the Jupyter notebook:
```bash
jupyter notebook risk_prediction_lightgbm.ipynb
```

2. The notebook will:
   - Generate synthetic data (or use your own data)
   - Perform exploratory data analysis
   - Apply SMOTEENN for balancing
   - Reduce dimensions with PCA
   - Optimize hyperparameters with Optuna
   - Train the final LightGBM model
   - Save the model and metadata

### Making Predictions

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('risk_prediction_model.pkl')

# Prepare your data (must have same features as training)
# data = pd.DataFrame(...)

# Make predictions
probabilities = model.predict_proba(data)[:, 1]
predictions = model.predict(data)
```

## Model Details

### Input Features
- **Total features**: 82
- **Numerical features**: 72
- **Categorical features**: 10

### Pipeline Components
1. **SMOTEENN**: Synthetic Minority Over-sampling with Edited Nearest Neighbors
2. **StandardScaler**: Normalizes numerical features
3. **PCA**: Reduces numerical features while preserving 95% variance
4. **LightGBM Classifier**: Gradient boosting with optimized hyperparameters

### Performance Metrics
The model is evaluated using:
- AUC-ROC Score
- Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation with TimeSeriesSplit

## Customization

### Using Your Own Data

Replace the data generation section with your data loading code:

```python
# Instead of generate_imbalanced_data()
df = pd.read_csv('your_data.csv')

# Ensure you have:
# - A 'target' column (binary: 0 or 1)
# - Numerical and categorical features
# - A time-based column for TimeSeriesSplit (optional)
```

### Adjusting Parameters

Key parameters to tune:
- `n_components` in PCA (default: 0.95)
- `n_splits` in TimeSeriesSplit (default: 5)
- `n_trials` in Optuna optimization (default: 30)

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Acknowledgments

This implementation uses several open-source libraries:
- LightGBM by Microsoft
- Optuna for hyperparameter optimization
- imbalanced-learn for SMOTEENN
- scikit-learn for preprocessing and evaluation
