# Risk Prediction Model with LightGBM

A comprehensive risk prediction model implementation using advanced machine learning techniques for imbalanced time series data.

## 🌟 Features

- **LightGBM** - High-performance gradient boosting framework
- **SMOTEENN** - Combines SMOTE oversampling with Edited Nearest Neighbours undersampling for handling imbalanced data
- **PCA** - Principal Component Analysis for dimensionality reduction 
- **TimeSeriesSplit** - Proper cross-validation for time series data
- **Optuna** - Bayesian hyperparameter optimization
- **Comprehensive Evaluation** - Multiple metrics and visualizations

## 📊 Dataset Specifications

- **Total Features**: 82 (72 numerical + 10 categorical)
- **Target**: Binary risk classification (imbalanced ~15% positive class)
- **Structure**: Time series ordered data
- **Sample Size**: 10,000 records (configurable)

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
python risk_prediction_model.py
```

## 📁 File Structure

```
├── risk_prediction_model.py    # Main implementation script
├── risk_prediction_model.ipynb # Jupyter notebook version
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── lightgbm_risk_model.joblib  # Saved model artifacts (generated)
├── model_evaluation_plots.png  # Evaluation visualizations (generated)
└── pca_analysis.png           # PCA analysis plots (generated)
```

## 🔧 Implementation Details

### 1. Data Generation
Simulates a realistic financial risk dataset with:
- **Financial features**: income, debt-to-income ratio, credit utilization, payment history
- **Demographic features**: employment type, education, marital status, housing
- **Time-dependent patterns**: seasonal trends and temporal correlations
- **Realistic risk factors**: Based on domain knowledge of financial risk

### 2. Data Preprocessing
- **Label Encoding**: Converts categorical variables to numerical
- **Missing Value Handling**: Median imputation for numerical, mode for categorical
- **Feature Identification**: Automatic separation of numerical and categorical features

### 3. Time Series Split
- **Temporal Ordering**: Maintains chronological order (no random shuffling)
- **80/20 Split**: First 80% for training, last 20% for testing
- **Distribution Preservation**: Maintains target distribution across splits

### 4. Feature Scaling & PCA
- **StandardScaler**: Normalizes numerical features to zero mean, unit variance
- **PCA**: Reduces dimensionality while preserving 95% of variance
- **Component Analysis**: Detailed analysis of principal components

### 5. Imbalanced Data Handling (SMOTEENN)
- **SMOTE**: Synthetic Minority Oversampling Technique
- **ENN**: Edited Nearest Neighbours for cleaning
- **Combined Approach**: First oversample minorities, then clean boundaries

### 6. Hyperparameter Optimization (Optuna)
- **Bayesian Optimization**: TPE (Tree-structured Parzen Estimator) sampler
- **Pruning**: MedianPruner for early stopping of unpromising trials
- **CV Integration**: Uses TimeSeriesSplit for robust parameter selection
- **50 Trials**: Balances optimization time vs. performance

### 7. Model Training (LightGBM)
- **Gradient Boosting**: Fast and efficient implementation
- **Imbalanced Support**: Built-in handling for imbalanced datasets
- **Early Stopping**: Prevents overfitting
- **Feature Importance**: Gain-based importance calculation

### 8. Cross-Validation
- **TimeSeriesSplit**: 5-fold time-aware cross-validation
- **Multiple Metrics**: AUC, Accuracy, Precision, Recall, F1-Score
- **Comprehensive Reporting**: Mean and standard deviation for all metrics

### 9. Model Evaluation
- **Classification Metrics**: Complete performance assessment
- **ROC & PR Curves**: Visual performance analysis
- **Confusion Matrix**: Detailed classification breakdown
- **Feature Importance**: Top contributing features
- **Prediction Distribution**: Risk score distributions by class

### 10. Visualization & Analysis
- **Model Performance**: ROC curves, precision-recall, confusion matrix
- **Feature Analysis**: PCA component analysis, feature importance
- **Distribution Analysis**: Prediction probability distributions
- **Cross-Validation**: Performance consistency across folds

## 📈 Expected Performance

Based on the simulated dataset, typical performance metrics:

- **AUC-ROC**: 0.85-0.95
- **Accuracy**: 0.80-0.90
- **Precision**: 0.70-0.85
- **Recall**: 0.75-0.90
- **F1-Score**: 0.70-0.85

*Note: Performance may vary based on data characteristics and random seed*

## 🎯 Model Usage

### Making Predictions on New Data

```python
from risk_prediction_model import predict_risk
import pandas as pd

# Load your new data (must have same structure as training)
new_data = pd.read_csv('new_risk_data.csv')

# Make predictions
risk_probabilities, binary_predictions = predict_risk(new_data)

# Results
for i, (prob, pred) in enumerate(zip(risk_probabilities, binary_predictions)):
    print(f"Sample {i+1}: Risk Probability = {prob:.3f}, Classification = {'High Risk' if pred else 'Low Risk'}")
```

### Loading Saved Model

```python
import joblib

# Load all model artifacts
artifacts = joblib.load('lightgbm_risk_model.joblib')

model = artifacts['model']
scaler = artifacts['scaler']
pca = artifacts['pca']
label_encoders = artifacts['label_encoders']
# ... other components
```

## 🔍 Key Features Explained

### Why SMOTEENN?
- **SMOTE**: Generates synthetic samples for minority class
- **ENN**: Removes noise and improves class boundaries
- **Combined**: Better than either technique alone for imbalanced data

### Why PCA?
- **Dimensionality Reduction**: 82 → ~40-50 features (preserving 95% variance)
- **Noise Reduction**: Removes redundant information
- **Computational Efficiency**: Faster training and prediction

### Why TimeSeriesSplit?
- **Temporal Integrity**: Respects time ordering in cross-validation
- **Realistic Evaluation**: Simulates real-world deployment scenario
- **Avoids Data Leakage**: Future data never used to predict past

### Why Optuna?
- **Efficient Optimization**: Bayesian approach outperforms grid/random search
- **Pruning**: Saves computational time by stopping poor trials early
- **Flexible**: Easy to add/remove hyperparameters

## 📊 Model Artifacts

The saved model includes:
- **Trained LightGBM model**
- **Fitted preprocessors** (scaler, PCA, label encoders)
- **SMOTEENN transformer**
- **Best hyperparameters**
- **Feature metadata**
- **Performance metrics**
- **Cross-validation results**

## 🚀 Advanced Usage

### Custom Dataset
Replace the `generate_risk_dataset()` function with your data loading:

```python
def load_your_dataset():
    df = pd.read_csv('your_dataset.csv')
    # Ensure you have a 'target' column and 'date' column
    return df

# In main(), replace:
# df = generate_risk_dataset(...)
df = load_your_dataset()
```

### Hyperparameter Tuning
Modify the Optuna objective function to tune different parameters:

```python
# In optimize_hyperparameters(), modify the params dict:
params = {
    'num_leaves': trial.suggest_int('num_leaves', 10, 300),  # Wider range
    'max_depth': trial.suggest_int('max_depth', 3, 15),     # Add max depth
    # ... other parameters
}
```

### Different Algorithms
The pipeline can be adapted for other algorithms:

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Replace LightGBM sections with your preferred algorithm
```

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce dataset size or PCA components
2. **Long Training Time**: Reduce Optuna trials or CV folds
3. **Poor Performance**: Check data quality and feature engineering
4. **Convergence Issues**: Adjust LightGBM learning rate

### Performance Optimization

1. **Parallel Processing**: Set `n_jobs=-1` in applicable functions
2. **GPU Support**: Use LightGBM GPU version if available
3. **Memory Management**: Use categorical features instead of label encoding
4. **Early Stopping**: Tune early stopping patience

## 📚 Dependencies

Core requirements:
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning utilities
- `lightgbm>=3.3.0` - Gradient boosting framework
- `imbalanced-learn>=0.8.0` - Imbalanced data handling
- `optuna>=3.0.0` - Hyperparameter optimization
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualizations

## 🤝 Contributing

Feel free to improve this implementation:
1. Fork the repository
2. Create feature branch
3. Make improvements
4. Submit pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue with detailed description

---

**Note**: This implementation uses simulated data. For production use, replace with your actual dataset and validate all preprocessing steps for your specific domain.
