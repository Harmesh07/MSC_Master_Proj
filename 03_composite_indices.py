#!/usr/bin/env python3
"""
Composite Indices Creation and Reliability Analysis
Privacy, Security, and Compliance of GenAI in LMS: DPIA Study

Module: 7150CEM MSc Data Science Project
Date: 28 November 2025
Author: Harmesh Deshwal

This script creates the four composite indices as specified in the research design.
"""

# %% [markdown]
# # Composite Indices Creation and Reliability Analysis
# Creates: PCI, DPAI, PBI, Data_Sharing_Score

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.random.seed(42)
sns.set_style('whitegrid')
sns.set_palette('Set2')

print("✓ Libraries loaded")

# %% [markdown]
# ## 1. Load Cleaned Dataset

# %%
df = pd.read_csv('../data_proc/survey_clean.csv')
column_ref = pd.read_csv('../data_proc/column_reference.csv')

print(f"Dataset: {df.shape[0]:,} students, {df.shape[1]} variables")
print(f"Column reference: {len(column_ref)} variable mappings")

# %% [markdown]
# ## 2. Privacy Concern Index (PCI)
# Mean of Q22a-j (10 ethical concern items)

# %%
pci_items = ['Q22a', 'Q22b', 'Q22c', 'Q22d', 'Q22e', 'Q22f', 'Q22g', 'Q22h', 'Q22i', 'Q22j']
available_pci = [col for col in pci_items if col in df.columns]
print(f"PCI items available: {len(available_pci)}/10")

df['PCI'] = df[available_pci].mean(axis=1)

print("\n=== PRIVACY CONCERN INDEX (PCI) ===")
print(f"Mean:   {df['PCI'].mean():.3f}")
print(f"Median: {df['PCI'].median():.3f}")
print(f"SD:     {df['PCI'].std():.3f}")
print(f"Min:    {df['PCI'].min():.3f}")
print(f"Max:    {df['PCI'].max():.3f}")
print(f"Valid N: {df['PCI'].notna().sum():,}")

# %%
def cronbach_alpha(df, items):
    """Calculate Cronbach's Alpha for reliability analysis."""
    df_items = df[items].dropna()
    k = len(items)
    item_variances = df_items.var(axis=0, ddof=1)
    total_variance = df_items.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

pci_alpha = cronbach_alpha(df, available_pci)
print(f"\nCronbach's Alpha (PCI): {pci_alpha:.3f}")
if pci_alpha >= 0.9:
    print("✓ Excellent internal consistency")
elif pci_alpha >= 0.8:
    print("✓ Good internal consistency")

# %% [markdown]
# ## 3. Data Protection Awareness Index (DPAI)
# Mean of Q21a-d + Q23d

# %%
dpai_items = ['Q21a', 'Q21b', 'Q21c', 'Q21d', 'Q23d']
available_dpai = [col for col in dpai_items if col in df.columns]
print(f"DPAI items available: {len(available_dpai)}/5")

df['DPAI'] = df[available_dpai].mean(axis=1)

print("\n=== DATA PROTECTION AWARENESS INDEX (DPAI) ===")
print(f"Mean:   {df['DPAI'].mean():.3f}")
print(f"Median: {df['DPAI'].median():.3f}")
print(f"SD:     {df['DPAI'].std():.3f}")
print(f"Valid N: {df['DPAI'].notna().sum():,}")

dpai_alpha = cronbach_alpha(df, available_dpai)
print(f"\nCronbach's Alpha (DPAI): {dpai_alpha:.3f}")

# %% [markdown]
# ## 4. Perceived Benefit Index (PBI)
# Mean of Q26a-j + Q27a-j (20 items)

# %%
pbi_items = [
    'Q26a', 'Q26b', 'Q26c', 'Q26d', 'Q26e', 'Q26f', 'Q26g', 'Q26h', 'Q26i', 'Q26j',
    'Q27a', 'Q27b', 'Q27c', 'Q27d', 'Q27e', 'Q27f', 'Q27g', 'Q27h', 'Q27i', 'Q27j'
]
available_pbi = [col for col in pbi_items if col in df.columns]
print(f"PBI items available: {len(available_pbi)}/20")

df['PBI'] = df[available_pbi].mean(axis=1)

print("\n=== PERCEIVED BENEFIT INDEX (PBI) ===")
print(f"Mean:   {df['PBI'].mean():.3f}")
print(f"Median: {df['PBI'].median():.3f}")
print(f"SD:     {df['PBI'].std():.3f}")
print(f"Valid N: {df['PBI'].notna().sum():,}")

pbi_alpha = cronbach_alpha(df, available_pbi)
print(f"\nCronbach's Alpha (PBI): {pbi_alpha:.3f}")

# %% [markdown]
# ## 5. Data Sharing Score
# Count of AI tools used (Q13a-f)

# %%
sharing_items = ['Q13a', 'Q13b', 'Q13c', 'Q13d', 'Q13e', 'Q13f']
available_sharing = [col for col in sharing_items if col in df.columns]
print(f"Data Sharing items available: {len(available_sharing)}/6")

for col in available_sharing:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0).astype(int)

df['Data_Sharing_Score'] = df[available_sharing].sum(axis=1)

print("\n=== DATA SHARING SCORE ===")
print(f"Mean:   {df['Data_Sharing_Score'].mean():.3f}")
print(f"Median: {df['Data_Sharing_Score'].median():.0f}")
print(f"SD:     {df['Data_Sharing_Score'].std():.3f}")

print("\nTool usage breakdown:")
tool_names = ['ChatGPT', 'Copilot', 'Gemini', 'Perplexity', 'Claude', 'Other']
for col, name in zip(available_sharing, tool_names):
    usage_pct = (df[col].sum() / len(df)) * 100
    print(f"  {name}: {usage_pct:.1f}%")

# %% [markdown]
# ## 6. Create Categorical Variables

# %%
pci_33 = df['PCI'].quantile(0.33)
pci_67 = df['PCI'].quantile(0.67)

df['PCI_Category'] = pd.cut(df['PCI'], 
                              bins=[0, pci_33, pci_67, 5],
                              labels=['Low', 'Moderate', 'High'],
                              include_lowest=True)

print("\nPCI Category Distribution:")
print(df['PCI_Category'].value_counts())

# %% [markdown]
# ## 7. Prepare Demographics

# %%
# Gender
if 'Q2' in df.columns:
    df['Gender'] = df['Q2']
    print("Gender distribution:")
    print(df['Gender'].value_counts())

# Age
if 'Q3' in df.columns:
    df['Age'] = pd.to_numeric(df['Q3'], errors='coerce')
    df.loc[(df['Age'] < 16) | (df['Age'] > 99), 'Age'] = np.nan
    print(f"\nAge: Mean = {df['Age'].mean():.1f}, SD = {df['Age'].std():.1f}")
    
    df['Age_Group'] = pd.cut(df['Age'], 
                              bins=[0, 21, 25, 30, 40, 100],
                              labels=['18-21', '22-25', '26-30', '31-40', '41+'])

# Study Level
if 'Q8' in df.columns:
    df['Study_Level'] = df['Q8']

# Policy Awareness
if 'Q20' in df.columns:
    df['Policy_Aware'] = df['Q20']
    print("\nAI Policy Awareness:")
    print(df['Policy_Aware'].value_counts())

# %% [markdown]
# ## 8. Save Enhanced Dataset

# %%
output_file = '../data_proc/survey_with_indices.csv'
df.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print("COMPOSITE INDICES SUMMARY")
print('='*80)
print(f"\n1. Privacy Concern Index (PCI)")
print(f"   Mean: {df['PCI'].mean():.3f}, SD: {df['PCI'].std():.3f}, α: {pci_alpha:.3f}")
print(f"\n2. Data Protection Awareness Index (DPAI)")
print(f"   Mean: {df['DPAI'].mean():.3f}, SD: {df['DPAI'].std():.3f}, α: {dpai_alpha:.3f}")
print(f"\n3. Perceived Benefit Index (PBI)")
print(f"   Mean: {df['PBI'].mean():.3f}, SD: {df['PBI'].std():.3f}, α: {pbi_alpha:.3f}")
print(f"\n4. Data Sharing Score")
print(f"   Mean: {df['Data_Sharing_Score'].mean():.3f}, Median: {df['Data_Sharing_Score'].median():.0f}")
print(f"\n✓ Enhanced dataset saved: {output_file}")
print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print('='*80)


