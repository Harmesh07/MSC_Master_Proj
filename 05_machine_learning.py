#!/usr/bin/env python3
"""
Machine Learning Analysis - Week 2
Privacy, Security, and Compliance of GenAI in LMS: DPIA Study

Module: 7150CEM MSc Data Science Project
Date: November 2025
Author: Harmesh Deshwal

Performs:
1. Random Forest classification to predict PCI level
2. Feature importance analysis
3. K-Means clustering for student segmentation
"""

# %% [markdown]
# # Machine Learning Analysis
# Week 2: Random Forest + K-Means Clustering

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score, precision_recall_fscore_support,
                              roc_curve, auc, roc_auc_score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("✓ Libraries loaded")

# %% [markdown]
# ## 1. Load Dataset and Prepare Features

# %%
df = pd.read_csv('../data_proc/survey_with_indices.csv')
print(f"Dataset: {df.shape[0]:,} students, {df.shape[1]} variables")

# Select features for modeling
feature_cols = []
if 'Age' in df.columns:
    feature_cols.append('Age')
if 'Gender' in df.columns:
    feature_cols.append('Gender')
if 'Study_Level' in df.columns:
    feature_cols.append('Study_Level')
if 'DPAI' in df.columns:
    feature_cols.append('DPAI')
if 'Data_Sharing_Score' in df.columns:
    feature_cols.append('Data_Sharing_Score')
if 'Policy_Aware' in df.columns:
    feature_cols.append('Policy_Aware')
if 'PBI' in df.columns:
    feature_cols.append('PBI')

print(f"\nFeatures selected: {feature_cols}")

# Target variable
if 'PCI_Category' in df.columns:
    target = 'PCI_Category'
    print(f"Target: {target}")
    print(f"\nTarget distribution:")
    print(df[target].value_counts())
else:
    print("⚠ PCI_Category not found, creating it...")
    pci_33 = df['PCI'].quantile(0.33)
    pci_67 = df['PCI'].quantile(0.67)
    df['PCI_Category'] = pd.cut(df['PCI'], 
                                  bins=[0, pci_33, pci_67, 5],
                                  labels=['Low', 'Moderate', 'High'],
                                  include_lowest=True)
    target = 'PCI_Category'

# %% [markdown]
# ## 2. Data Preprocessing

# %%
# Create modeling dataset
df_ml = df[feature_cols + [target]].copy()

# Remove missing values
df_ml = df_ml.dropna()
print(f"\nModeling dataset after removing NAs: {len(df_ml):,} cases")

# Encode categorical variables
le_target = LabelEncoder()
df_ml[target + '_encoded'] = le_target.fit_transform(df_ml[target])

# Encode categorical features
categorical_features = []
for col in feature_cols:
    if df_ml[col].dtype == 'object' or df_ml[col].nunique() < 10:
        categorical_features.append(col)
        le = LabelEncoder()
        df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))

# Create feature matrix
X_cols = [col + '_encoded' if col in categorical_features else col 
          for col in feature_cols]

# Ensure all features are numeric
X = df_ml[[c for c in X_cols if c in df_ml.columns]].copy()

# Handle any remaining non-numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(X.median())

y = df_ml[target + '_encoded']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Classes: {le_target.classes_}")

# %% [markdown]
# ## 3. Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print(f"Split ratio: 70% train / 30% test")

# %% [markdown]
# ## 4. Random Forest Model

# %%
print("\n" + "="*80)
print("RANDOM FOREST CLASSIFICATION")
print("="*80)

# Train model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf_model.fit(X_train, y_train)
print("✓ Model trained")

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance:")
print(f"  Training Accuracy: {train_accuracy:.3f}")
print(f"  Test Accuracy: {test_accuracy:.3f}")

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation:")
print(f"  Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

# %% [markdown]
# ## 5. Classification Report

# %%
print("\n" + "="*80)
print("CLASSIFICATION REPORT (Test Set)")
print("="*80)

print("\n" + classification_report(
    y_test, y_pred_test,
    target_names=le_target.classes_,
    digits=3
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# Calculate per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_test, average=None
)

metrics_df = pd.DataFrame({
    'Class': le_target.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})
print("\n" + metrics_df.to_string(index=False))

# %% [markdown]
# ## 6. Feature Importance Analysis

# %%
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop predictors of Privacy Concern:")
print(feature_importance.to_string(index=False))

# Save feature importance
import os
os.makedirs('../results', exist_ok=True)
feature_importance.to_csv('../results/feature_importance.csv', index=False)
print("\n✓ Feature importance saved")

# Interpretation
print("\n### INTERPRETATION ###")
top_feature = feature_importance.iloc[0]
print(f"\nMost important predictor: {top_feature['Feature']}")
print(f"Importance score: {top_feature['Importance']:.3f}")

if 'DPAI' in str(top_feature['Feature']):
    print("  • Data protection awareness is the strongest predictor")
    print("  • Students who understand regulations have different concern levels")
elif 'Data_Sharing' in str(top_feature['Feature']):
    print("  • Data sharing behavior is the strongest predictor")
    print("  • Actual behavior matters more than demographics")
elif 'PBI' in str(top_feature['Feature']):
    print("  • Perceived benefits are the strongest predictor")
    print("  • Risk-benefit perception drives privacy concern")

# %% [markdown]
# ## 7. K-Means Clustering for Student Segmentation

# %%
print("\n" + "="*80)
print("K-MEANS CLUSTERING")
print("="*80)

# Prepare clustering features
cluster_features = []
if 'PCI' in df.columns:
    cluster_features.append('PCI')
if 'DPAI' in df.columns:
    cluster_features.append('DPAI')
if 'Data_Sharing_Score' in df.columns:
    cluster_features.append('Data_Sharing_Score')
if 'PBI' in df.columns:
    cluster_features.append('PBI')

print(f"\nClustering features: {cluster_features}")

df_cluster = df[cluster_features].dropna()
print(f"Cases for clustering: {len(df_cluster):,}")

# Standardize features
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df_cluster)

# Determine optimal k using elbow method
print("\nDetermining optimal number of clusters...")
inertias = []
silhouette_scores = []
K_range = range(2, 7)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))

print("\nCluster quality metrics:")
for k, inertia, sil in zip(K_range, inertias, silhouette_scores):
    print(f"  k={k}: Inertia={inertia:.0f}, Silhouette={sil:.3f}")

# Select optimal k (typically 3-4 for interpretability)
optimal_k = 4
print(f"\nUsing k={optimal_k} clusters for final model")

# %% [markdown]
# ## 8. Final K-Means Model

# %%
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_cluster)

# Add cluster labels to dataframe
df_cluster['Cluster'] = cluster_labels

# Cluster statistics
print("\n" + "="*80)
print("CLUSTER PROFILES")
print("="*80)

cluster_profiles = df_cluster.groupby('Cluster').agg({
    'PCI': ['mean', 'std', 'count'],
    'DPAI': ['mean', 'std'] if 'DPAI' in cluster_features else None,
    'Data_Sharing_Score': ['mean', 'std'] if 'Data_Sharing_Score' in cluster_features else None,
    'PBI': ['mean', 'std'] if 'PBI' in cluster_features else None
}).round(3)

print("\n" + cluster_profiles.to_string())

# Name clusters based on characteristics
print("\n### CLUSTER INTERPRETATIONS ###")
for cluster_id in range(optimal_k):
    cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
    pci_mean = cluster_data['PCI'].mean()
    dpai_mean = cluster_data['DPAI'].mean() if 'DPAI' in cluster_features else 0
    sharing_mean = cluster_data['Data_Sharing_Score'].mean() if 'Data_Sharing_Score' in cluster_features else 0
    
    print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
    print(f"  PCI: {pci_mean:.2f}, DPAI: {dpai_mean:.2f}, Sharing: {sharing_mean:.2f}")
    
    # Characterize cluster
    if pci_mean > df_cluster['PCI'].quantile(0.67):
        if dpai_mean > df_cluster['DPAI'].quantile(0.67) if 'DPAI' in cluster_features else False:
            print(f"  → 'Informed & Concerned': High awareness + High concern")
            print(f"     Policy: Provide privacy-enhancing tools")
        else:
            print(f"  → 'Worried': High concern, lower awareness")
            print(f"     Policy: Education on regulations and rights")
    elif pci_mean < df_cluster['PCI'].quantile(0.33):
        if sharing_mean > df_cluster['Data_Sharing_Score'].quantile(0.67) if 'Data_Sharing_Score' in cluster_features else False:
            print(f"  → 'Unconcerned Sharers': Low concern + High sharing")
            print(f"     Policy: Awareness campaigns on risks")
        else:
            print(f"  → 'Low Risk': Low concern + Limited sharing")
            print(f"     Policy: Maintain current practices")
    else:
        print(f"  → 'Moderate': Average across dimensions")
        print(f"     Policy: General best practices")

# Save clustering results
df_cluster.to_csv('../results/cluster_assignments.csv', index=False)
print("\n✓ Cluster assignments saved")

# %% [markdown]
# ## 9. Model Evaluation Summary

# %%
print("\n" + "="*80)
print("MACHINE LEARNING SUMMARY")
print("="*80)

ml_summary = {
    'Model': ['Random Forest', 'K-Means'],
    'Primary_Metric': [f'Accuracy: {test_accuracy:.3f}', 
                       f'Silhouette: {silhouette_scores[optimal_k-2]:.3f}'],
    'Key_Finding': [
        f"Top predictor: {feature_importance.iloc[0]['Feature']}",
        f"{optimal_k} distinct student segments identified"
    ]
}

ml_summary_df = pd.DataFrame(ml_summary)
print("\n" + ml_summary_df.to_string(index=False))

ml_summary_df.to_csv('../results/ml_summary.csv', index=False)

print("\n" + "="*80)
print("MACHINE LEARNING ANALYSIS COMPLETE")
print("="*80)
print("\n✓ Random Forest classification completed")
print("✓ Feature importance identified")
print("✓ Student segmentation completed")
print("✓ Results saved to results/ directory")
print("\nKey contributions:")
print("  1. Predictive model for privacy concern levels")
print("  2. Feature importance rankings for policy focus")
print("  3. Student segments requiring different interventions")
print("\nProceed to DPIA Risk Assessment")
print("="*80)


