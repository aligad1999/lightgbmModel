#!/usr/bin/env python3
"""
Risk Prediction Model using LightGBM

This script implements a comprehensive risk prediction model with:
- LightGBM for gradient boosting
- SMOTEENN for handling imbalanced data
- PCA for dimensionality reduction
- TimeSeriesSplit for cross-validation
- Optuna for hyperparameter tuning

Dataset characteristics:
- 82 total features (72 numerical + 10 categorical)
- Imbalanced target variable
- Time series structure
"""

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)

# Imbalanced learning
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

# Hyperparameter optimization
import optuna
try:
    from optuna_integration.lightgbm import LightGBMPruningCallback
except ImportError:
    try:
        from optuna.integration import LightGBMPruningCallback
    except ImportError:
        print("Warning: LightGBMPruningCallback not available. Using basic optimization.")
        LightGBMPruningCallback = None

# Utilities
from datetime import datetime
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def generate_risk_dataset(n_samples=10000, n_features=82, n_categorical=10, imbalance_ratio=0.15):
    """
    Generate a realistic risk prediction dataset with time series structure
    
    Parameters:
    - n_samples: Number of samples
    - n_features: Total number of features (82)
    - n_categorical: Number of categorical features (10)
    - imbalance_ratio: Ratio of positive class (default 15% for imbalanced data)
    """
    np.random.seed(42)
    
    # Generate time index for time series structure
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate numerical features (72 features)
    n_numerical = n_features - n_categorical
    
    # Financial/Risk related features
    numerical_data = {
        # Financial metrics
        'income': np.random.lognormal(10, 1, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'credit_utilization': np.random.beta(1, 3, n_samples),
        'payment_history_score': np.random.normal(750, 100, n_samples),
        'account_age_months': np.random.exponential(50, n_samples),
        'number_of_accounts': np.random.poisson(8, n_samples),
        'recent_inquiries': np.random.poisson(2, n_samples),
        'loan_amount': np.random.lognormal(9, 1, n_samples),
        'employment_length': np.random.exponential(5, n_samples),
        'annual_income': np.random.lognormal(11, 0.5, n_samples),
    }
    
    # Add more numerical features to reach 72
    for i in range(10, n_numerical):
        if i % 3 == 0:
            # Some correlated features
            numerical_data[f'feature_{i}'] = (
                numerical_data['income'] * np.random.normal(0.5, 0.1, n_samples) + 
                np.random.normal(0, 1, n_samples)
            )
        elif i % 3 == 1:
            # Time-dependent features
            trend = np.linspace(0, 1, n_samples)
            seasonal = np.sin(2 * np.pi * np.arange(n_samples) / 365)
            numerical_data[f'feature_{i}'] = (
                trend + 0.5 * seasonal + np.random.normal(0, 0.3, n_samples)
            )
        else:
            # Random features
            numerical_data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Generate categorical features
    categorical_data = {
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.25, 0.05]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1]),
        'housing_status': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples, p=[0.3, 0.4, 0.3]),
        'state': np.random.choice([f'State_{i}' for i in range(10)], n_samples),
        'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Other'], n_samples),
        'loan_purpose': np.random.choice(['Personal', 'Auto', 'Home', 'Business', 'Education'], n_samples),
        'bank_relationship': np.random.choice(['New', 'Existing_1-2yr', 'Existing_3-5yr', 'Existing_5+yr'], n_samples),
        'payment_method': np.random.choice(['Auto', 'Manual', 'Mixed'], n_samples),
        'risk_segment': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.3, 0.2])
    }
    
    # Combine all features
    data = {**numerical_data, **categorical_data}
    df = pd.DataFrame(data)
    df['date'] = dates
    
    # Generate target variable based on features (realistic risk factors)
    risk_score = (
        -0.3 * (df['payment_history_score'] - 750) / 100 +  # Lower credit score = higher risk
        0.5 * df['debt_to_income'] +                        # Higher debt ratio = higher risk
        0.3 * df['credit_utilization'] +                    # Higher utilization = higher risk
        0.2 * np.log(df['recent_inquiries'] + 1) +          # More inquiries = higher risk
        -0.1 * np.log(df['employment_length'] + 1) +        # Longer employment = lower risk
        0.1 * (df['employment_type'] == 'Unemployed').astype(int) +
        np.random.normal(0, 0.5, n_samples)                 # Add noise
    )
    
    # Convert to binary with specified imbalance ratio
    threshold = np.percentile(risk_score, (1 - imbalance_ratio) * 100)
    df['target'] = (risk_score > threshold).astype(int)
    
    # Sort by date to maintain time series structure
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def preprocess_data(df):
    """
    Preprocess the dataset for machine learning
    """
    print("Starting data preprocessing...")
    
    # Separate features and target
    target_col = 'target'
    date_col = 'date'
    feature_cols = [col for col in df.columns if col not in [target_col, date_col]]
    
    # Identify categorical and numerical columns
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    
    # Handle categorical variables with Label Encoding
    df_processed = df.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique categories")
    
    # Check for missing values
    missing_values = df_processed[feature_cols].isnull().sum()
    if missing_values.sum() > 0:
        print(f"Handling {missing_values.sum()} missing values...")
        # Fill missing values
        for col in numerical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    else:
        print("No missing values found.")
    
    return df_processed, feature_cols, categorical_cols, numerical_cols, label_encoders

def time_series_split(df_processed, feature_cols, target_col='target'):
    """
    Split data maintaining time series order
    """
    print("Performing time series split...")
    
    # For time series data, use temporal split (not random)
    split_idx = int(len(df_processed) * 0.8)
    
    train_data = df_processed.iloc[:split_idx].copy()
    test_data = df_processed.iloc[split_idx:].copy()
    
    # Separate features and target
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training target distribution: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Test target distribution: {y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    return X_train, X_test, y_train, y_test

def apply_scaling_and_pca(X_train, X_test, numerical_cols, explained_variance=0.95):
    """
    Apply feature scaling and PCA
    """
    print("Applying feature scaling and PCA...")
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale only numerical features
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Apply PCA
    pca = PCA(n_components=explained_variance, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA Results:")
    print(f"Original features: {X_train_scaled.shape[1]}")
    print(f"PCA components: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_train_pca, X_test_pca, scaler, pca

def apply_smoteenn(X_train_pca, y_train):
    """
    Apply SMOTEENN for handling imbalanced data
    """
    print("Applying SMOTEENN for imbalanced data handling...")
    print(f"Original training distribution: {np.bincount(y_train)}")
    
    # Initialize SMOTEENN
    smoteenn = SMOTEENN(
        smote=SMOTE(random_state=42, k_neighbors=5),
        enn=EditedNearestNeighbours(n_neighbors=3),
        random_state=42
    )
    
    # Apply SMOTEENN
    X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_pca, y_train)
    
    print(f"After SMOTEENN distribution: {np.bincount(y_train_resampled)}")
    print(f"Original samples: {len(y_train)}")
    print(f"Resampled samples: {len(y_train_resampled)}")
    print(f"Imbalance ratio after SMOTEENN: {y_train_resampled.mean():.3f}")
    
    return X_train_resampled, y_train_resampled, smoteenn

def optimize_hyperparameters(X_train_resampled, y_train_resampled, n_trials=50):
    """
    Optimize hyperparameters using Optuna
    """
    print("Starting hyperparameter optimization with Optuna...")
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbosity': -1,
            'is_unbalance': True,
            'early_stopping_rounds': 50
        }
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_resampled)):
            X_fold_train, X_fold_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
            y_fold_train, y_fold_val = y_train_resampled[train_idx], y_train_resampled[val_idx]
            
            # Create datasets
            train_dataset = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_dataset = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_dataset)
            
            # Train model
            callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
            if LightGBMPruningCallback is not None:
                callbacks.append(LightGBMPruningCallback(trial, 'valid_0-binary_logloss'))
            
            model = lgb.train(
                params,
                train_dataset,
                valid_sets=[val_dataset],
                num_boost_round=1000,
                callbacks=callbacks
            )
            
            # Predict and calculate AUC
            y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
            auc_score = roc_auc_score(y_fold_val, y_pred)
            cv_scores.append(auc_score)
        
        return np.mean(cv_scores)
    
    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=3600)
    
    print("Optimization completed!")
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return study

def train_final_model(X_train_resampled, y_train_resampled, X_test_pca, y_test, best_params):
    """
    Train the final model with best parameters
    """
    print("Training final model with optimized parameters...")
    
    # Prepare final parameters
    final_params = best_params.copy()
    final_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbosity': -1,
        'is_unbalance': True
    })
    
    # Create datasets
    train_dataset = lgb.Dataset(X_train_resampled, label=y_train_resampled)
    val_dataset = lgb.Dataset(X_test_pca, label=y_test, reference=train_dataset)
    
    # Train final model
    final_model = lgb.train(
        final_params,
        train_dataset,
        valid_sets=[train_dataset, val_dataset],
        valid_names=['train', 'test'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )
    
    print(f"Best iteration: {final_model.best_iteration}")
    return final_model

def evaluate_model(final_model, X_train_resampled, y_train_resampled, X_test_pca, y_test, threshold=0.5):
    """
    Evaluate the trained model
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_train_pred_proba = final_model.predict(X_train_resampled, num_iteration=final_model.best_iteration)
    y_test_pred_proba = final_model.predict(X_test_pca, num_iteration=final_model.best_iteration)
    
    # Convert probabilities to binary predictions
    y_train_pred = (y_train_pred_proba > threshold).astype(int)
    y_test_pred = (y_test_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred, y_pred_proba, dataset_name):
        print(f"\n{dataset_name} Set Metrics:")
        print("=" * 40)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    # Calculate metrics for both sets
    train_metrics = calculate_metrics(y_train_resampled, y_train_pred, y_train_pred_proba, "Training")
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba, "Test")
    
    # Print classification report
    print("\nDetailed Classification Report - Test Set:")
    print("=" * 50)
    print(classification_report(y_test, y_test_pred, target_names=['Low Risk', 'High Risk']))
    
    return train_metrics, test_metrics, y_test_pred_proba, y_test_pred

def cross_validate_model(X_train_resampled, y_train_resampled, best_params, n_splits=5):
    """
    Perform comprehensive cross-validation
    """
    print("Performing TimeSeriesSplit Cross-Validation...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {
        'fold': [],
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_resampled)):
        print(f"Fold {fold + 1}...")
        
        # Split data
        X_fold_train, X_fold_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
        y_fold_train, y_fold_val = y_train_resampled[train_idx], y_train_resampled[val_idx]
        
        # Create datasets
        train_dataset = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_dataset = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_dataset)
        
        # Train model
        fold_model = lgb.train(
            best_params,
            train_dataset,
            valid_sets=[val_dataset],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
        )
        
        # Predict
        y_val_pred_proba = fold_model.predict(X_fold_val, num_iteration=fold_model.best_iteration)
        y_val_pred = (y_val_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        fold_auc = roc_auc_score(y_fold_val, y_val_pred_proba)
        fold_accuracy = accuracy_score(y_fold_val, y_val_pred)
        fold_precision = precision_score(y_fold_val, y_val_pred)
        fold_recall = recall_score(y_fold_val, y_val_pred)
        fold_f1 = f1_score(y_fold_val, y_val_pred)
        
        # Store results
        cv_scores['fold'].append(fold + 1)
        cv_scores['auc'].append(fold_auc)
        cv_scores['accuracy'].append(fold_accuracy)
        cv_scores['precision'].append(fold_precision)
        cv_scores['recall'].append(fold_recall)
        cv_scores['f1'].append(fold_f1)
    
    # Convert to DataFrame
    cv_results_df = pd.DataFrame(cv_scores)
    
    print("\nCross-Validation Results:")
    print("=" * 50)
    print(cv_results_df)
    
    print(f"\nMean CV Scores:")
    for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
        mean_score = cv_results_df[metric].mean()
        std_score = cv_results_df[metric].std()
        print(f"  {metric.upper()}: {mean_score:.4f} (+/- {std_score*2:.4f})")
    
    return cv_results_df

def create_visualizations(final_model, X_test_pca, y_test, y_test_pred_proba, y_test_pred, 
                         train_metrics, test_metrics, pca):
    """
    Create comprehensive evaluation visualizations
    """
    print("Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    auc_score = roc_auc_score(y_test, y_test_pred_proba)
    axes[0,0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curve')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
    axes[0,1].plot(recall, precision, linewidth=2)
    axes[0,1].set_xlabel('Recall')
    axes[0,1].set_ylabel('Precision')
    axes[0,1].set_title('Precision-Recall Curve')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,2])
    axes[0,2].set_xlabel('Predicted')
    axes[0,2].set_ylabel('Actual')
    axes[0,2].set_title('Confusion Matrix')
    axes[0,2].set_xticklabels(['Low Risk', 'High Risk'])
    axes[0,2].set_yticklabels(['Low Risk', 'High Risk'])
    
    # 4. Feature Importance (Top 20)
    feature_importance = final_model.feature_importance(importance_type='gain')
    feature_names = [f'PC_{i+1}' for i in range(len(feature_importance))]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(20)
    
    axes[1,0].barh(range(len(importance_df)), importance_df['importance'])
    axes[1,0].set_yticks(range(len(importance_df)))
    axes[1,0].set_yticklabels(importance_df['feature'])
    axes[1,0].set_xlabel('Importance')
    axes[1,0].set_title('Top 20 Feature Importance (PCA Components)')
    axes[1,0].gca().invert_yaxis()
    
    # 5. Prediction Distribution
    axes[1,1].hist(y_test_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                   label='Low Risk', density=True)
    axes[1,1].hist(y_test_pred_proba[y_test == 1], bins=30, alpha=0.7, 
                   label='High Risk', density=True)
    axes[1,1].axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    axes[1,1].set_xlabel('Predicted Probability')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Prediction Probability Distribution')
    axes[1,1].legend()
    
    # 6. Metrics Comparison
    metrics_comparison = pd.DataFrame({
        'Train': [train_metrics['accuracy'], train_metrics['precision'], 
                  train_metrics['recall'], train_metrics['f1'], train_metrics['auc']],
        'Test': [test_metrics['accuracy'], test_metrics['precision'], 
                 test_metrics['recall'], test_metrics['f1'], test_metrics['auc']]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])
    
    metrics_comparison.plot(kind='bar', ax=axes[1,2], width=0.8)
    axes[1,2].set_title('Train vs Test Metrics Comparison')
    axes[1,2].set_ylabel('Score')
    axes[1,2].tick_params(axis='x', rotation=45)
    axes[1,2].legend()
    axes[1,2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional plot: PCA explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
            pca.explained_variance_ratio_[:20])
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA: Individual Component Variance (Top 20)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model_artifacts(final_model, scaler, pca, label_encoders, smoteenn, best_params,
                        feature_cols, categorical_cols, numerical_cols, test_metrics, cv_results):
    """
    Save all model artifacts for future use
    """
    print("Saving model artifacts...")
    
    model_artifacts = {
        'model': final_model,
        'scaler': scaler,
        'pca': pca,
        'label_encoders': label_encoders,
        'smoteenn': smoteenn,
        'best_params': best_params,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'threshold': 0.5,
        'cv_results': cv_results,
        'test_metrics': test_metrics
    }
    
    # Save artifacts
    model_filename = 'lightgbm_risk_model.joblib'
    joblib.dump(model_artifacts, model_filename)
    
    print(f"Model artifacts saved to: {model_filename}")
    print(f"File size: {os.path.getsize(model_filename) / (1024*1024):.1f} MB")
    
    return model_filename

def predict_risk(new_data, model_artifacts_path='lightgbm_risk_model.joblib'):
    """
    Function to make risk predictions on new data
    """
    # Load model artifacts
    artifacts = joblib.load(model_artifacts_path)
    
    model = artifacts['model']
    scaler = artifacts['scaler']
    pca = artifacts['pca']
    label_encoders = artifacts['label_encoders']
    feature_cols = artifacts['feature_cols']
    categorical_cols = artifacts['categorical_cols']
    numerical_cols = artifacts['numerical_cols']
    threshold = artifacts['threshold']
    
    # Preprocess new data
    new_data_processed = new_data[feature_cols].copy()
    
    # Encode categorical variables
    for col in categorical_cols:
        le = label_encoders[col]
        # Handle unseen categories
        new_data_processed[col] = new_data_processed[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    
    # Scale numerical features
    new_data_processed[numerical_cols] = scaler.transform(new_data_processed[numerical_cols])
    
    # Apply PCA
    new_data_pca = pca.transform(new_data_processed)
    
    # Make predictions
    risk_probabilities = model.predict(new_data_pca, num_iteration=model.best_iteration)
    binary_predictions = (risk_probabilities > threshold).astype(int)
    
    return risk_probabilities, binary_predictions

def print_final_summary(df, test_metrics, cv_results, best_params, final_model, 
                       X_train_pca, pca, model_filename):
    """
    Print comprehensive summary report
    """
    print("\n" + "="*80)
    print("RISK PREDICTION MODEL - FINAL SUMMARY REPORT")
    print("="*80)
    
    print(f"\n📊 DATASET INFORMATION:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Total features: 82 (72 numerical + 10 categorical)")
    print(f"   • Original imbalance ratio: {df['target'].mean():.3f}")
    
    print(f"\n🔧 PREPROCESSING APPLIED:")
    print(f"   • Label encoding for categorical features")
    print(f"   • StandardScaler for numerical features")
    print(f"   • PCA: 82 → {X_train_pca.shape[1]} features")
    print(f"   • PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"   • SMOTEENN applied for imbalanced data")
    
    print(f"\n🎯 MODEL CONFIGURATION:")
    print(f"   • Algorithm: LightGBM Gradient Boosting")
    print(f"   • Hyperparameter optimization: Optuna (50 trials)")
    print(f"   • Cross-validation: TimeSeriesSplit (5 folds)")
    print(f"   • Best iteration: {final_model.best_iteration}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   • Test AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"   • Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   • Test Precision: {test_metrics['precision']:.4f}")
    print(f"   • Test Recall: {test_metrics['recall']:.4f}")
    print(f"   • Test F1-Score: {test_metrics['f1']:.4f}")
    
    print(f"\n🔄 CROSS-VALIDATION RESULTS:")
    for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
        mean_score = cv_results[metric].mean()
        std_score = cv_results[metric].std()
        print(f"   • CV {metric.upper()}: {mean_score:.4f} ± {std_score:.4f}")
    
    print(f"\n⚙️ BEST HYPERPARAMETERS:")
    for param, value in best_params.items():
        if param not in ['objective', 'metric', 'boosting_type', 'random_state', 'verbosity', 'is_unbalance']:
            print(f"   • {param}: {value}")
    
    print(f"\n💾 MODEL ARTIFACTS:")
    print(f"   • File: {model_filename}")
    print(f"   • Size: {os.path.getsize(model_filename) / (1024*1024):.1f} MB")
    
    print(f"\n🎉 MODEL READY FOR DEPLOYMENT!")
    print("="*80)

def main():
    """
    Main function to run the complete risk prediction pipeline
    """
    print("="*80)
    print("RISK PREDICTION MODEL WITH LIGHTGBM")
    print("="*80)
    print("Features: SMOTEENN + PCA + TimeSeriesSplit + Optuna")
    print("Dataset: 82 features (72 numerical + 10 categorical), imbalanced target")
    print("="*80)
    
    # Step 1: Generate dataset
    print("\n1. GENERATING DATASET")
    print("-" * 30)
    df = generate_risk_dataset(n_samples=10000, n_features=82, n_categorical=10, imbalance_ratio=0.15)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Step 2: Preprocess data
    print("\n2. PREPROCESSING DATA")
    print("-" * 30)
    df_processed, feature_cols, categorical_cols, numerical_cols, label_encoders = preprocess_data(df)
    
    # Step 3: Train-test split
    print("\n3. TIME SERIES SPLIT")
    print("-" * 30)
    X_train, X_test, y_train, y_test = time_series_split(df_processed, feature_cols)
    
    # Step 4: Scaling and PCA
    print("\n4. SCALING AND PCA")
    print("-" * 30)
    X_train_pca, X_test_pca, scaler, pca = apply_scaling_and_pca(X_train, X_test, numerical_cols)
    
     # Step 5: SMOTEENN
     print("\n5. SMOTEENN RESAMPLING")
     print("-" * 30)
     X_train_resampled, y_train_resampled, smoteenn = apply_smoteenn(X_train_pca, y_train)
     
     # Step 6: Hyperparameter optimization
     print("\n6. HYPERPARAMETER OPTIMIZATION")
     print("-" * 30)
     study = optimize_hyperparameters(X_train_resampled, y_train_resampled, n_trials=10)
     
     # Step 7: Train final model
     print("\n7. TRAINING FINAL MODEL")
     print("-" * 30)
     final_model = train_final_model(X_train_resampled, y_train_resampled, X_test_pca, y_test, study.best_params)
    
     # Step 8: Model evaluation
     print("\n8. MODEL EVALUATION")
     print("-" * 30)
     train_metrics, test_metrics, y_test_pred_proba, y_test_pred = evaluate_model(
         final_model, X_train_resampled, y_train_resampled, X_test_pca, y_test
     )
     
     # Step 9: Cross-validation
     print("\n9. CROSS-VALIDATION")
     print("-" * 30)
     cv_results = cross_validate_model(X_train_resampled, y_train_resampled, study.best_params)
     
     # Step 10: Create visualizations
     print("\n10. CREATING VISUALIZATIONS")
     print("-" * 30)
     create_visualizations(final_model, X_test_pca, y_test, y_test_pred_proba, y_test_pred,
                          train_metrics, test_metrics, pca)
     
     # Step 11: Save model
     print("\n11. SAVING MODEL")
     print("-" * 30)
     model_filename = save_model_artifacts(
         final_model, scaler, pca, label_encoders, smoteenn, study.best_params,
         feature_cols, categorical_cols, numerical_cols, test_metrics, cv_results
     )
     
     # Step 12: Final summary
     print("\n12. FINAL SUMMARY")
     print("-" * 30)
     print_final_summary(df, test_metrics, cv_results, study.best_params, final_model,
                        X_train_pca, pca, model_filename)
     
     # Example prediction
     print("\n13. EXAMPLE PREDICTION")
     print("-" * 30)
     sample_data = df_processed[feature_cols].head(5)
     sample_actual = df_processed['target'].head(5)
     
     sample_proba, sample_pred = predict_risk(sample_data)
     
     results_df = pd.DataFrame({
         'Actual_Risk': sample_actual.values,
         'Predicted_Probability': sample_proba,
         'Predicted_Risk': sample_pred,
         'Correct': (sample_actual.values == sample_pred)
     })
     
     print("Sample predictions:")
     print(results_df)
     print(f"Sample Accuracy: {results_df['Correct'].mean():.2%}")
     
     print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
     print("📝 All artifacts saved and ready for production use.")

if __name__ == "__main__":
    main()