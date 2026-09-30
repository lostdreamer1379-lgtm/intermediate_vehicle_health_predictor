# Vehicle Health Monitoring - Predictive Maintenance ML Model
# Complete Colab Notebook Code
# Train with cross-validation and export for local use

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: DATA LOADING & EXPLORATION")
print("=" * 80)

# Load dataset
df = pd.read_csv('dataset.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nBasic statistics:")
print(df.describe())
print(f"\nTarget distribution:")
print(df['Failure'].value_counts())
print(f"Failure rate: {df['Failure'].mean()*100:.2f}%")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("=" * 80)

# Separate features and target
X = df.drop(['Record_ID', 'Failure'], axis=1)
y = df['Failure']

feature_names = X.columns.tolist()
print(f"\nFeatures used: {feature_names}")
print(f"\nFeature count: {len(feature_names)}")

# Check for missing values
print(f"\nMissing values: {X.isnull().sum().sum()}")

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: TRAIN-TEST SPLIT (80-20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set failure rate: {y_train.mean()*100:.2f}%")
print(f"Test set failure rate: {y_test.mean()*100:.2f}%")

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FEATURE SCALING (StandardScaler)")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")
print(f"Training set - Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
print(f"Test set - Mean: {X_test_scaled.mean():.4f}, Std: {X_test_scaled.std():.4f}")

# ============================================================================
# STEP 5: CROSS-VALIDATION SETUP
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: STRATIFIED K-FOLD CROSS-VALIDATION (k=5)")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Using 5-fold stratified cross-validation for unbiased evaluation")

# ============================================================================
# STEP 6: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: MODEL TRAINING & EVALUATION")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

cv_results = {}
best_model_name = None
best_cv_score = -1

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=skf, scoring='roc_auc', n_jobs=-1
    )
    
    cv_results[model_name] = {
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"Cross-validation ROC-AUC Scores (5-fold): {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_test = model.predict(X_test_scaled)
    y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_auc = roc_auc_score(y_train, y_pred_proba_train)
    test_auc = roc_auc_score(y_test, y_pred_proba_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    
    cv_results[model_name]['model'] = model
    cv_results[model_name]['train_acc'] = train_acc
    cv_results[model_name]['test_acc'] = test_acc
    cv_results[model_name]['train_auc'] = train_auc
    cv_results[model_name]['test_auc'] = test_auc
    cv_results[model_name]['test_f1'] = test_f1
    cv_results[model_name]['test_precision'] = test_precision
    cv_results[model_name]['test_recall'] = test_recall
    cv_results[model_name]['y_pred_test'] = y_pred_test
    cv_results[model_name]['y_pred_proba_test'] = y_pred_proba_test
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training ROC-AUC: {train_auc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['No Failure', 'Failure']))
    
    # Track best model by CV score
    if cv_scores.mean() > best_cv_score:
        best_cv_score = cv_scores.mean()
        best_model_name = model_name

# ============================================================================
# STEP 7: MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL COMPARISON SUMMARY")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'CV ROC-AUC Mean': [cv_results[m]['cv_mean'] for m in cv_results.keys()],
    'CV ROC-AUC Std': [cv_results[m]['cv_std'] for m in cv_results.keys()],
    'Train Accuracy': [cv_results[m]['train_acc'] for m in cv_results.keys()],
    'Test Accuracy': [cv_results[m]['test_acc'] for m in cv_results.keys()],
    'Test ROC-AUC': [cv_results[m]['test_auc'] for m in cv_results.keys()],
    'Test F1-Score': [cv_results[m]['test_f1'] for m in cv_results.keys()],
})

print("\n" + comparison_df.to_string(index=False))
print(f"\n✓ Best Model (by CV ROC-AUC): {best_model_name}")
print(f"  CV Score: {cv_results[best_model_name]['cv_mean']:.4f} (+/- {cv_results[best_model_name]['cv_std']:.4f})")

# ============================================================================
# STEP 8: BEST MODEL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: BEST MODEL DETAILED ANALYSIS")
print("=" * 80)

best_model = cv_results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")

# Confusion Matrix
cm = confusion_matrix(y_test, cv_results[best_model_name]['y_pred_test'])
print(f"\nConfusion Matrix:")
print(cm)
print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 5 Important Features:")
    print(importance_df.head())

# ============================================================================
# STEP 9: SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: SAVING MODEL AND ARTIFACTS")
print("=" * 80)

# Save best model
model_filename = 'best_vehicle_health_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Best model saved: {model_filename}")

# Save scaler
scaler_filename = 'feature_scaler.pkl'
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved: {scaler_filename}")

# Save feature names
feature_filename = 'feature_names.pkl'
with open(feature_filename, 'wb') as f:
    pickle.dump(feature_names, f)
print(f"✓ Feature names saved: {feature_filename}")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'feature_names': feature_names,
    'cv_mean_score': cv_results[best_model_name]['cv_mean'],
    'cv_std_score': cv_results[best_model_name]['cv_std'],
    'test_accuracy': cv_results[best_model_name]['test_acc'],
    'test_roc_auc': cv_results[best_model_name]['test_auc'],
    'test_f1': cv_results[best_model_name]['test_f1'],
    'test_precision': cv_results[best_model_name]['test_precision'],
    'test_recall': cv_results[best_model_name]['test_recall'],
}

metadata_filename = 'model_metadata.pkl'
with open(metadata_filename, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Model metadata saved: {metadata_filename}")

# Create results summary CSV
results_summary = pd.DataFrame([{
    'Model': best_model_name,
    'CV_ROC_AUC': cv_results[best_model_name]['cv_mean'],
    'CV_Std': cv_results[best_model_name]['cv_std'],
    'Test_Accuracy': cv_results[best_model_name]['test_acc'],
    'Test_ROC_AUC': cv_results[best_model_name]['test_auc'],
    'Test_F1': cv_results[best_model_name]['test_f1'],
    'Test_Precision': cv_results[best_model_name]['test_precision'],
    'Test_Recall': cv_results[best_model_name]['test_recall'],
}])

results_summary.to_csv('model_results_summary.csv', index=False)
print(f"✓ Results summary saved: model_results_summary.csv")

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC Curve
ax = axes[0, 0]
for model_name, results in cv_results.items():
    fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba_test'])
    auc = roc_auc_score(y_test, results['y_pred_proba_test'])
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison')
ax.legend()
ax.grid(alpha=0.3)

# Confusion Matrix for best model
ax = axes[0, 1]
cm_best = confusion_matrix(y_test, cv_results[best_model_name]['y_pred_test'])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Confusion Matrix - {best_model_name}')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# Model Comparison - CV Scores
ax = axes[1, 0]
model_names = list(cv_results.keys())
cv_means = [cv_results[m]['cv_mean'] for m in model_names]
cv_stds = [cv_results[m]['cv_std'] for m in model_names]
ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
ax.set_ylabel('ROC-AUC Score')
ax.set_title('Cross-Validation ROC-AUC Comparison')
ax.set_ylim([0.5, 1.0])
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)

# Test Metrics for best model
ax = axes[1, 1]
metrics = ['Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']
values = [
    cv_results[best_model_name]['test_acc'],
    cv_results[best_model_name]['test_auc'],
    cv_results[best_model_name]['test_f1'],
    cv_results[best_model_name]['test_precision'],
    cv_results[best_model_name]['test_recall'],
]
ax.bar(metrics, values, alpha=0.7, color='steelblue')
ax.set_ylabel('Score')
ax.set_title(f'Test Metrics - {best_model_name}')
ax.set_ylim([0, 1])
for i, v in enumerate(values):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: model_evaluation_results.png")
plt.show()

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\nFiles generated:")
print("  1. best_vehicle_health_model.pkl (Trained model)")
print("  2. feature_scaler.pkl (Feature scaler)")
print("  3. feature_names.pkl (Feature names)")
print("  4. model_metadata.pkl (Model metadata)")
print("  5. model_results_summary.csv (Results summary)")
print("  6. model_evaluation_results.png (Visualizations)")
print("\nThese files can be downloaded and used locally!")
