#!/usr/bin/env python3
"""
Risk Prediction Model using LightGBM with 4.4% Class Imbalance

This script implements a comprehensive risk prediction model with:
- LightGBM for gradient boosting
- SMOTEENN for handling imbalanced data (4.4% positive class)
- PCA for dimensionality reduction
- TimeSeriesSplit for cross-validation
- Optuna for hyperparameter tuning

Dataset characteristics:
- 82 total features (72 numerical + 10 categorical)
- Highly imbalanced target variable (4.4% positive class)
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
from sklearn.model_selection import TimeSeriesSplit
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

# Utilities
from datetime import datetime
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_risk_dataset(n_samples=10000, n_features=82, n_categorical=10, imbalance_ratio=0.044):
    """
    Generate a realistic risk prediction dataset with time series structure
    Updated to use 4.4% imbalance ratio for highly imbalanced scenario
    """
    np.random.seed(42)
    
    # Generate time index for time series structure
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate numerical features (72 features)
    n_numerical = n_features - n_categorical
    
    # Financial/Risk related features
    numerical_data = {
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
            numerical_data[f'feature_{i}'] = (
                numerical_data['income'] * np.random.normal(0.5, 0.1, n_samples) + 
                np.random.normal(0, 1, n_samples)
            )
        elif i % 3 == 1:
            trend = np.linspace(0, 1, n_samples)
            seasonal = np.sin(2 * np.pi * np.arange(n_samples) / 365)
            numerical_data[f'feature_{i}'] = (
                trend + 0.5 * seasonal + np.random.normal(0, 0.3, n_samples)
            )
        else:
            numerical_data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Generate categorical features
    categorical_data = {
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                          n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                          n_samples, p=[0.3, 0.4, 0.25, 0.05]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 
                                         n_samples, p=[0.4, 0.5, 0.1]),
        'housing_status': np.random.choice(['Rent', 'Own', 'Mortgage'], 
                                         n_samples, p=[0.3, 0.4, 0.3]),
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
    
    # Generate target variable with stronger risk factors for 4.4% imbalance
    risk_score = (
        -0.4 * (df['payment_history_score'] - 750) / 100 +  # Stronger credit score impact
        0.6 * df['debt_to_income'] +                        # Higher debt ratio impact
        0.4 * df['credit_utilization'] +                    # Higher utilization impact
        0.3 * np.log(df['recent_inquiries'] + 1) +          # More inquiries impact
        -0.2 * np.log(df['employment_length'] + 1) +        # Employment stability
        0.3 * (df['employment_type'] == 'Unemployed').astype(int) +
        0.2 * (df['risk_segment'] == 'High').astype(int) +  # Risk segment impact
        np.random.normal(0, 0.4, n_samples)                 # Reduced noise for clearer signal
    )
    
    # Convert to binary with 4.4% imbalance ratio
    threshold = np.percentile(risk_score, (1 - imbalance_ratio) * 100)
    df['target'] = (risk_score > threshold).astype(int)
    
    # Sort by date to maintain time series structure
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def main():
    """Main function to run the complete risk prediction pipeline"""
    
    print("=" * 80)
    print("RISK PREDICTION MODEL WITH LIGHTGBM")
    print("=" * 80)
    print("Features: SMOTEENN + PCA + TimeSeriesSplit + Optuna")
    print("Dataset: 82 features, 4.4% class imbalance (highly imbalanced)")
    print("=" * 80)
    
    # Step 1: Generate dataset with 4.4% imbalance
    print("\n1. GENERATING DATASET (4.4% IMBALANCE)")
    print("-" * 50)
    df = generate_risk_dataset(n_samples=10000, n_features=82, n_categorical=10, imbalance_ratio=0.044)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"Imbalance ratio: {df['target'].mean():.3f} ({df['target'].mean()*100:.1f}%)")
    
    # Step 2: Basic preprocessing
    print("\n2. PREPROCESSING DATA")
    print("-" * 50)
    
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
    
    # Step 3: Time series split
    print("\n3. TIME SERIES SPLIT")
    print("-" * 50)
    
    split_idx = int(len(df_processed) * 0.8)
    train_data = df_processed.iloc[:split_idx].copy()
    test_data = df_processed.iloc[split_idx:].copy()
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training target distribution: {y_train.value_counts(normalize=True).round(4).to_dict()}")
    print(f"Test target distribution: {y_test.value_counts(normalize=True).round(4).to_dict()}")
    
    # Step 4: Feature scaling and PCA
    print("\n4. SCALING AND PCA")
    print("-" * 50)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Original features: {X_train_scaled.shape[1]}")
    print(f"PCA components: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Step 5: Handle imbalanced data with SMOTEENN
    print("\n5. SMOTEENN RESAMPLING")
    print("-" * 50)
    
    print(f"Original training distribution: {np.bincount(y_train)}")
    print(f"Original imbalance ratio: {y_train.mean():.4f}")
    
    # Initialize SMOTEENN for highly imbalanced data
    smoteenn = SMOTEENN(
        smote=SMOTE(random_state=42, k_neighbors=3),  # Reduced neighbors for sparse minority class
        enn=EditedNearestNeighbours(n_neighbors=3),
        random_state=42
    )
    
    # Apply SMOTEENN
    X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_pca, y_train)
    
    print(f"After SMOTEENN distribution: {np.bincount(y_train_resampled)}")
    print(f"Original samples: {len(y_train)}")
    print(f"Resampled samples: {len(y_train_resampled)}")
    print(f"New imbalance ratio: {y_train_resampled.mean():.4f}")
    
    # Step 6: Hyperparameter optimization with Optuna
    print("\n6. HYPERPARAMETER OPTIMIZATION")
    print("-" * 50)
    
    def objective(trial):
        # Suggest hyperparameters optimized for imbalanced data
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
            'random_state': 42,
            'verbosity': -1,
            'is_unbalance': True,
            'scale_pos_weight': len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])
        }
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced for speed
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train_resampled):
            X_fold_train, X_fold_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
            y_fold_train, y_fold_val = y_train_resampled[train_idx], y_train_resampled[val_idx]
            
            # Create datasets
            train_dataset = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_dataset = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_dataset)
            
            # Train model
            model = lgb.train(
                params,
                train_dataset,
                valid_sets=[val_dataset],
                num_boost_round=100,  # Reduced for speed
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            
            # Predict and calculate AUC
            y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
            auc_score = roc_auc_score(y_fold_val, y_pred)
            cv_scores.append(auc_score)
        
        return np.mean(cv_scores)
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20, timeout=300)  # 5 minutes max
    
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Step 7: Train final model
    print("\n7. TRAINING FINAL MODEL")
    print("-" * 50)
    
    # Prepare final parameters
    final_params = study.best_params.copy()
    final_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbosity': -1,
        'is_unbalance': True,
        'scale_pos_weight': len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])
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
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    print(f"Best iteration: {final_model.best_iteration}")
    
    # Step 8: Model evaluation
    print("\n8. MODEL EVALUATION")
    print("-" * 50)
    
    # Make predictions
    y_train_pred_proba = final_model.predict(X_train_resampled, num_iteration=final_model.best_iteration)
    y_test_pred_proba = final_model.predict(X_test_pca, num_iteration=final_model.best_iteration)
    
    # Optimize threshold for imbalanced data
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    # Use Youden's J statistic to find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")
    
    # Convert probabilities to binary predictions
    y_train_pred = (y_train_pred_proba > optimal_threshold).astype(int)
    y_test_pred = (y_test_pred_proba > optimal_threshold).astype(int)
    
    # Calculate comprehensive metrics
    def calculate_metrics(y_true, y_pred, y_pred_proba, dataset_name):
        print(f"\n{dataset_name} Set Metrics:")
        print("=" * 40)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
    
    # Calculate metrics for both sets
    train_metrics = calculate_metrics(y_train_resampled, y_train_pred, y_train_pred_proba, "Training")
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba, "Test")
    
    # Detailed classification report
    print("\nDetailed Classification Report - Test Set:")
    print("=" * 50)
    print(classification_report(y_test, y_test_pred, target_names=['Low Risk', 'High Risk'], zero_division=0))
    
    # Step 9: Create visualizations
    print("\n9. CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Set up plotting
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC Curve
    fpr_plot, tpr_plot, _ = roc_curve(y_test, y_test_pred_proba)
    auc_score = roc_auc_score(y_test, y_test_pred_proba)
    axes[0,0].plot(fpr_plot, tpr_plot, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0,0].scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                     label=f'Optimal Threshold = {optimal_threshold:.3f}')
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curve')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_pred_proba)
    axes[0,1].plot(recall_curve, precision_curve, linewidth=2)
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
    
    # 4. Feature Importance (Top 15)
    feature_importance = final_model.feature_importance(importance_type='gain')
    feature_names = [f'PC_{i+1}' for i in range(len(feature_importance))]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(15)
    
    axes[1,0].barh(range(len(importance_df)), importance_df['importance'])
    axes[1,0].set_yticks(range(len(importance_df)))
    axes[1,0].set_yticklabels(importance_df['feature'])
    axes[1,0].set_xlabel('Importance')
    axes[1,0].set_title('Top 15 Feature Importance (PCA Components)')
    axes[1,0].gca().invert_yaxis()
    
    # 5. Prediction Distribution
    axes[1,1].hist(y_test_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                   label='Low Risk', density=True, color='blue')
    axes[1,1].hist(y_test_pred_proba[y_test == 1], bins=30, alpha=0.7, 
                   label='High Risk', density=True, color='red')
    axes[1,1].axvline(x=optimal_threshold, color='green', linestyle='--', 
                     label=f'Optimal Threshold ({optimal_threshold:.3f})')
    axes[1,1].set_xlabel('Predicted Probability')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Prediction Probability Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('risk_model_evaluation.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'risk_model_evaluation.png'")
    plt.show()
    
    # Step 10: Save model artifacts
    print("\n10. SAVING MODEL ARTIFACTS")
    print("-" * 50)
    
    model_artifacts = {
        'model': final_model,
        'scaler': scaler,
        'pca': pca,
        'label_encoders': label_encoders,
        'smoteenn': smoteenn,
        'best_params': final_params,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'optimal_threshold': optimal_threshold,
        'test_metrics': test_metrics,
        'imbalance_ratio': 0.044
    }
    
    model_filename = 'lightgbm_risk_model_4_4_percent.joblib'
    joblib.dump(model_artifacts, model_filename)
    
    print(f"Model artifacts saved to: {model_filename}")
    print(f"File size: {os.path.getsize(model_filename) / (1024*1024):.1f} MB")
    
    # Step 11: Final summary
    print("\n11. FINAL SUMMARY")
    print("=" * 80)
    print("RISK PREDICTION MODEL - FINAL SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\n📊 DATASET INFORMATION:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Total features: 82 (72 numerical + 10 categorical)")
    print(f"   • Class imbalance ratio: 4.4% (highly imbalanced)")
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Test samples: {len(X_test):,}")
    
    print(f"\n🔧 PREPROCESSING APPLIED:")
    print(f"   • Label encoding for categorical features")
    print(f"   • StandardScaler for numerical features")
    print(f"   • PCA: 82 → {X_train_pca.shape[1]} features")
    print(f"   • PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"   • SMOTEENN applied for highly imbalanced data")
    print(f"   • Optimal threshold: {optimal_threshold:.4f} (Youden's J)")
    
    print(f"\n🎯 MODEL CONFIGURATION:")
    print(f"   • Algorithm: LightGBM Gradient Boosting")
    print(f"   • Hyperparameter optimization: Optuna (20 trials)")
    print(f"   • Cross-validation: TimeSeriesSplit (3 folds)")
    print(f"   • Best iteration: {final_model.best_iteration}")
    print(f"   • Scale pos weight: {final_params.get('scale_pos_weight', 1):.2f}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   • Test AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"   • Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   • Test Precision: {test_metrics['precision']:.4f}")
    print(f"   • Test Recall: {test_metrics['recall']:.4f}")
    print(f"   • Test F1-Score: {test_metrics['f1']:.4f}")
    
    print(f"\n⚙️ BEST HYPERPARAMETERS:")
    key_params = ['num_leaves', 'learning_rate', 'feature_fraction', 'min_child_samples']
    for param in key_params:
        if param in study.best_params:
            print(f"   • {param}: {study.best_params[param]}")
    
    print(f"\n💾 MODEL ARTIFACTS:")
    print(f"   • File: {model_filename}")
    print(f"   • Includes: Model, preprocessors, encoders, optimal threshold")
    
    print(f"\n🎉 MODEL READY FOR DEPLOYMENT!")
    print(f"📝 Optimized for 4.4% class imbalance scenario")
    print("=" * 80)
    
    # Example prediction function
    def predict_risk_example(sample_data):
        """Example function to make predictions on new data"""
        # This would be extracted to a separate module in production
        artifacts = joblib.load(model_filename)
        model = artifacts['model']
        scaler = artifacts['scaler']
        pca = artifacts['pca']
        threshold = artifacts['optimal_threshold']
        
        # Preprocess (simplified for demo)
        sample_scaled = sample_data.copy()
        sample_scaled[numerical_cols] = scaler.transform(sample_data[numerical_cols])
        sample_pca = pca.transform(sample_scaled)
        
        # Predict
        risk_proba = model.predict(sample_pca, num_iteration=model.best_iteration)
        risk_binary = (risk_proba > threshold).astype(int)
        
        return risk_proba, risk_binary
    
    print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📊 Model optimized for highly imbalanced data (4.4% positive class)")
    
    return model_artifacts

if __name__ == "__main__":
    main()