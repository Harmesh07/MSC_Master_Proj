#!/usr/bin/env python3
"""
Statistical Analysis - Week 1
Privacy, Security, and Compliance of GenAI in LMS: DPIA Study

Module: 7150CEM MSc Data Science Project
Date: 28 November 2025
Author: Harmesh Deshwal

Performs:
1. Descriptive statistics
2. Gender differences in PCI (t-test)
3. PCI × DPAI correlation
4. Privacy paradox test: PCI × Data_Sharing_Score correlation
"""

# %% [markdown]
# # Statistical Analysis
# Week 1: Focused statistical tests as per research plan

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

print("✓ Libraries loaded")

# %% [markdown]
# ## 1. Load Dataset with Indices

# %%
df = pd.read_csv('../data_proc/survey_with_indices.csv')
print(f"Dataset: {df.shape[0]:,} students, {df.shape[1]} variables")
print(f"\nComposite indices available:")
print(f"  • PCI: {df['PCI'].notna().sum():,} valid cases")
print(f"  • DPAI: {df['DPAI'].notna().sum():,} valid cases")
print(f"  • PBI: {df['PBI'].notna().sum():,} valid cases")
print(f"  • Data_Sharing_Score: {df['Data_Sharing_Score'].notna().sum():,} valid cases")

# %% [markdown]
# ## 2. Descriptive Statistics

# %%
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Demographics
print("\n### DEMOGRAPHICS ###")
if 'Gender' in df.columns:
    print("\nGender Distribution:")
    print(df['Gender'].value_counts())
    print(f"  Percentage: {(df['Gender'].value_counts(normalize=True) * 100).round(1)}")

if 'Age' in df.columns:
    print(f"\nAge Statistics:")
    print(f"  Mean: {df['Age'].mean():.1f}")
    print(f"  Median: {df['Age'].median():.1f}")
    print(f"  SD: {df['Age'].std():.1f}")
    print(f"  Range: {df['Age'].min():.0f} - {df['Age'].max():.0f}")

if 'Age_Group' in df.columns:
    print("\nAge Group Distribution:")
    print(df['Age_Group'].value_counts().sort_index())

if 'Study_Level' in df.columns:
    print("\nStudy Level Distribution:")
    print(df['Study_Level'].value_counts())

if 'Policy_Aware' in df.columns:
    print("\nInstitution Has AI Policy:")
    print(df['Policy_Aware'].value_counts())
    policy_yes_pct = (df['Policy_Aware'].value_counts(normalize=True) * 100)
    if 1 in policy_yes_pct.index:
        print(f"  % Aware of policy: {policy_yes_pct[1]:.1f}%")

# Composite Indices
print("\n### COMPOSITE INDICES ###")
indices = ['PCI', 'DPAI', 'PBI', 'Data_Sharing_Score']
index_desc = df[indices].describe()
print("\n" + index_desc.to_string())

# %% [markdown]
# ## 3. Test 1: Gender Differences in Privacy Concern

# %%
print("\n" + "="*80)
print("TEST 1: GENDER DIFFERENCES IN PRIVACY CONCERN (t-test)")
print("="*80)

if 'Gender' in df.columns and 'PCI' in df.columns:
    # Filter valid cases
    df_gender = df[df['Gender'].notna() & df['PCI'].notna()].copy()
    
    # Get unique gender codes
    gender_codes = df_gender['Gender'].unique()
    print(f"\nGender codes found: {gender_codes}")
    
    # Typically 1=Male, 2=Female (verify with data)
    if len(gender_codes) >= 2:
        male_code = sorted(gender_codes)[0]
        female_code = sorted(gender_codes)[1]
        
        male_pci = df_gender[df_gender['Gender'] == male_code]['PCI']
        female_pci = df_gender[df_gender['Gender'] == female_code]['PCI']
        
        print(f"\nMale PCI (n={len(male_pci)}): M={male_pci.mean():.3f}, SD={male_pci.std():.3f}")
        print(f"Female PCI (n={len(female_pci)}): M={female_pci.mean():.3f}, SD={female_pci.std():.3f}")
        print(f"Difference: {abs(female_pci.mean() - male_pci.mean()):.3f}")
        
        # Independent samples t-test
        t_stat, p_value = ttest_ind(female_pci.dropna(), male_pci.dropna())
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(male_pci)-1)*male_pci.std()**2 + 
                               (len(female_pci)-1)*female_pci.std()**2) / 
                              (len(male_pci) + len(female_pci) - 2))
        cohens_d = (female_pci.mean() - male_pci.mean()) / pooled_std
        
        print(f"\nt-test results:")
        print(f"  t = {t_stat:.3f}")
        print(f"  p = {p_value:.4f}")
        print(f"  Cohen's d = {cohens_d:.3f}")
        
        if p_value < 0.001:
            print(f"  ✓ Highly significant (p < .001)")
        elif p_value < 0.01:
            print(f"  ✓ Very significant (p < .01)")
        elif p_value < 0.05:
            print(f"  ✓ Significant (p < .05)")
        else:
            print(f"  ✗ Not significant (p ≥ .05)")
        
        if abs(cohens_d) >= 0.8:
            print(f"  Effect size: LARGE")
        elif abs(cohens_d) >= 0.5:
            print(f"  Effect size: MEDIUM")
        elif abs(cohens_d) >= 0.2:
            print(f"  Effect size: SMALL")
        else:
            print(f"  Effect size: NEGLIGIBLE")
else:
    print("\n⚠ Gender or PCI not available")

# %% [markdown]
# ## 4. Test 2: PCI × DPAI Correlation

# %%
print("\n" + "="*80)
print("TEST 2: PRIVACY CONCERN × DATA PROTECTION AWARENESS")
print("="*80)

if 'PCI' in df.columns and 'DPAI' in df.columns:
    df_corr = df[['PCI', 'DPAI']].dropna()
    
    # Pearson correlation
    r_pearson, p_pearson = pearsonr(df_corr['PCI'], df_corr['DPAI'])
    
    # Spearman correlation (for ordinal data)
    r_spearman, p_spearman = spearmanr(df_corr['PCI'], df_corr['DPAI'])
    
    print(f"\nPearson's r:")
    print(f"  r = {r_pearson:.3f}")
    print(f"  p = {p_pearson:.4f}")
    print(f"  n = {len(df_corr)}")
    
    print(f"\nSpearman's ρ (rho):")
    print(f"  ρ = {r_spearman:.3f}")
    print(f"  p = {p_spearman:.4f}")
    
    if p_pearson < 0.001:
        print(f"\n✓ Highly significant correlation (p < .001)")
    elif p_pearson < 0.05:
        print(f"\n✓ Significant correlation (p < .05)")
    else:
        print(f"\n✗ Not significant (p ≥ .05)")
    
    if abs(r_pearson) >= 0.5:
        print(f"Correlation strength: STRONG")
    elif abs(r_pearson) >= 0.3:
        print(f"Correlation strength: MODERATE")
    elif abs(r_pearson) >= 0.1:
        print(f"Correlation strength: WEAK")
    else:
        print(f"Correlation strength: NEGLIGIBLE")
    
    print(f"\nInterpretation:")
    if r_pearson > 0:
        print(f"  • Higher awareness is associated with higher privacy concern")
        print(f"  • Students who understand regulations are more concerned")
    else:
        print(f"  • Higher awareness is associated with lower privacy concern")
        print(f"  • May indicate awareness reduces worry")
else:
    print("\n⚠ PCI or DPAI not available")

# %% [markdown]
# ## 5. Test 3: Privacy Paradox (PCI × Data_Sharing_Score)

# %%
print("\n" + "="*80)
print("TEST 3: PRIVACY PARADOX")
print("Testing if high concern leads to low data sharing")
print("="*80)

if 'PCI' in df.columns and 'Data_Sharing_Score' in df.columns:
    df_paradox = df[['PCI', 'Data_Sharing_Score']].dropna()
    
    # Pearson correlation
    r_paradox, p_paradox = pearsonr(df_paradox['PCI'], df_paradox['Data_Sharing_Score'])
    
    # Spearman correlation
    r_spearman_paradox, p_spearman_paradox = spearmanr(df_paradox['PCI'], 
                                                         df_paradox['Data_Sharing_Score'])
    
    print(f"\nPearson's r:")
    print(f"  r = {r_paradox:.3f}")
    print(f"  p = {p_paradox:.4f}")
    print(f"  n = {len(df_paradox)}")
    
    print(f"\nSpearman's ρ:")
    print(f"  ρ = {r_spearman_paradox:.3f}")
    print(f"  p = {p_spearman_paradox:.4f}")
    
    print(f"\n### PRIVACY PARADOX INTERPRETATION ###")
    if p_paradox < 0.05:
        if r_paradox < -0.3:
            print("✓ NO PARADOX: Strong negative correlation")
            print("  Higher concern → Lower data sharing (expected behavior)")
        elif r_paradox < 0:
            print("✓ Weak negative correlation (expected direction)")
        elif r_paradox > 0:
            print("⚠ PRIVACY PARADOX DETECTED:")
            print("  Higher concern → Higher data sharing (paradoxical!)")
            print("  Students express concern but still share data")
    else:
        if abs(r_paradox) < 0.1:
            print("⚠ PRIVACY PARADOX EXISTS:")
            print("  No relationship between concern and sharing")
            print("  Privacy concerns do NOT predict behavior")
    
    # Additional analysis: high concern groups
    high_pci = df_paradox[df_paradox['PCI'] >= df_paradox['PCI'].quantile(0.75)]
    low_pci = df_paradox[df_paradox['PCI'] <= df_paradox['PCI'].quantile(0.25)]
    
    print(f"\nData Sharing by Concern Level:")
    print(f"  High PCI (top 25%): Mean sharing = {high_pci['Data_Sharing_Score'].mean():.2f}")
    print(f"  Low PCI (bottom 25%): Mean sharing = {low_pci['Data_Sharing_Score'].mean():.2f}")
    print(f"  Difference: {high_pci['Data_Sharing_Score'].mean() - low_pci['Data_Sharing_Score'].mean():.2f}")
else:
    print("\n⚠ PCI or Data_Sharing_Score not available")

# %% [markdown]
# ## 6. Additional Analysis: Age Group Differences

# %%
print("\n" + "="*80)
print("ADDITIONAL: AGE GROUP DIFFERENCES IN PCI")
print("="*80)

if 'Age_Group' in df.columns and 'PCI' in df.columns:
    df_age = df[df['Age_Group'].notna() & df['PCI'].notna()].copy()
    
    # Descriptive stats by age group
    print("\nPCI by Age Group:")
    age_pci_stats = df_age.groupby('Age_Group')['PCI'].agg(['count', 'mean', 'std'])
    print(age_pci_stats)
    
    # ANOVA
    age_groups = [group['PCI'].values for name, group in df_age.groupby('Age_Group')]
    f_stat, p_anova = f_oneway(*age_groups)
    
    print(f"\nOne-Way ANOVA:")
    print(f"  F = {f_stat:.3f}")
    print(f"  p = {p_anova:.4f}")
    
    if p_anova < 0.05:
        print(f"  ✓ Significant age group differences (p < .05)")
    else:
        print(f"  ✗ No significant age group differences")
else:
    print("\n⚠ Age_Group or PCI not available")

# %% [markdown]
# ## 7. Summary Statistics Table

# %%
print("\n" + "="*80)
print("CORRELATION MATRIX")
print("="*80)

corr_vars = ['PCI', 'DPAI', 'PBI', 'Data_Sharing_Score']
available_corr = [v for v in corr_vars if v in df.columns]

if len(available_corr) >= 2:
    corr_matrix = df[available_corr].corr()
    print("\nPearson Correlations:")
    print(corr_matrix.round(3))
    
    # Save correlation matrix
    corr_matrix.to_csv('../results/correlation_matrix.csv')
    print("\n✓ Correlation matrix saved to results/correlation_matrix.csv")

# %% [markdown]
# ## 8. Save Statistical Results

# %%
import os
os.makedirs('../results', exist_ok=True)

# Create results summary
results_summary = {
    'Test': ['Gender Differences', 'PCI×DPAI Correlation', 'Privacy Paradox'],
    'Statistic': [
        f't = {t_stat:.3f}' if 'Gender' in df.columns else 'N/A',
        f'r = {r_pearson:.3f}' if 'DPAI' in df.columns else 'N/A',
        f'r = {r_paradox:.3f}' if 'Data_Sharing_Score' in df.columns else 'N/A'
    ],
    'p_value': [
        f'{p_value:.4f}' if 'Gender' in df.columns else 'N/A',
        f'{p_pearson:.4f}' if 'DPAI' in df.columns else 'N/A',
        f'{p_paradox:.4f}' if 'Data_Sharing_Score' in df.columns else 'N/A'
    ],
    'Effect_Size': [
        f'd = {cohens_d:.3f}' if 'Gender' in df.columns else 'N/A',
        'r (itself)' if 'DPAI' in df.columns else 'N/A',
        'r (itself)' if 'Data_Sharing_Score' in df.columns else 'N/A'
    ],
    'Significant': [
        'Yes' if ('Gender' in df.columns and p_value < 0.05) else 'No/N/A',
        'Yes' if ('DPAI' in df.columns and p_pearson < 0.05) else 'No/N/A',
        'Yes' if ('Data_Sharing_Score' in df.columns and p_paradox < 0.05) else 'No/N/A'
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('../results/statistical_tests_summary.csv', index=False)

print("\n" + "="*80)
print("STATISTICAL ANALYSIS COMPLETE")
print("="*80)
print("\n✓ Statistical tests completed")
print("✓ Results saved to results/ directory")
print("\nKey findings:")
print("  1. Gender differences in privacy concern tested")
print("  2. Awareness-concern relationship examined")
print("  3. Privacy paradox investigated")
print("\nProceed to Week 2: Machine Learning Analysis")
print("="*80)


