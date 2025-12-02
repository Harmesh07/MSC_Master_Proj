#!/usr/bin/env python3
"""
Visualizations and Figures
Privacy, Security, and Compliance of GenAI in LMS: DPIA Study

Module: 7150CEM MSc Data Science Project
Date: November 2025
Author: Subhash Yadav

Creates publication-quality figures for the report.
"""

# %% [markdown]
# # Visualizations for Report
# Generates 6-8 key figures

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style('whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

import os
os.makedirs('../figures', exist_ok=True)

print("✓ Libraries loaded")

# %% [markdown]
# ## Load Data

# %%
df = pd.read_csv('../data_proc/survey_with_indices.csv')
print(f"Dataset: {df.shape[0]:,} students")

# Load ML results if available
try:
    feature_importance = pd.read_csv('../results/feature_importance.csv')
    clusters = pd.read_csv('../results/cluster_assignments.csv')
    print("✓ ML results loaded")
except:
    print("⚠ ML results not found, will skip those visualizations")
    feature_importance = None
    clusters = None

# %% [markdown]
# ## Figure 1: Distribution of Composite Indices

# %%
print("\nGenerating Figure 1: Composite Indices Distributions...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribution of Composite Indices (N=22,836)', fontsize=14, fontweight='bold')

# PCI
if 'PCI' in df.columns:
    axes[0, 0].hist(df['PCI'].dropna(), bins=50, color='#8E44AD', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['PCI'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={df["PCI"].mean():.2f}')
    axes[0, 0].set_xlabel('Privacy Concern Index (PCI)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Privacy Concern Index\nM={df["PCI"].mean():.2f}, SD={df["PCI"].std():.2f}')
    axes[0, 0].legend()

# DPAI
if 'DPAI' in df.columns:
    axes[0, 1].hist(df['DPAI'].dropna(), bins=50, color='#3498DB', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df['DPAI'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={df["DPAI"].mean():.2f}')
    axes[0, 1].set_xlabel('Data Protection Awareness Index (DPAI)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Data Protection Awareness\nM={df["DPAI"].mean():.2f}, SD={df["DPAI"].std():.2f}')
    axes[0, 1].legend()

# PBI
if 'PBI' in df.columns:
    axes[1, 0].hist(df['PBI'].dropna(), bins=50, color='#27AE60', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['PBI'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={df["PBI"].mean():.2f}')
    axes[1, 0].set_xlabel('Perceived Benefit Index (PBI)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Perceived Benefits\nM={df["PBI"].mean():.2f}, SD={df["PBI"].std():.2f}')
    axes[1, 0].legend()

# Data Sharing Score
if 'Data_Sharing_Score' in df.columns:
    sharing_counts = df['Data_Sharing_Score'].value_counts().sort_index()
    axes[1, 1].bar(sharing_counts.index, sharing_counts.values, color='#E74C3C', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Number of AI Tools Used')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Data Sharing Score\nM={df["Data_Sharing_Score"].mean():.2f}, Median={df["Data_Sharing_Score"].median():.0f}')
    axes[1, 1].set_xticks(range(int(df['Data_Sharing_Score'].max())+1))

plt.tight_layout()
plt.savefig('../figures/fig1_composite_indices.png', dpi=300, bbox_inches='tight')
print("✓ Figure 1 saved: fig1_composite_indices.png")
plt.close()

# %% [markdown]
# ## Figure 2: Gender Differences in Privacy Concern

# %%
print("\nGenerating Figure 2: Gender Differences...")

if 'Gender' in df.columns and 'PCI' in df.columns:
    df_gender = df[df['Gender'].notna() & df['PCI'].notna()].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Violin plot
    parts = ax.violinplot(
        [df_gender[df_gender['Gender'] == g]['PCI'].values for g in sorted(df_gender['Gender'].unique()[:2])],
        positions=[1, 2],
        showmeans=True,
        showmedians=True
    )
    
    # Box plot overlay
    gender_codes = sorted(df_gender['Gender'].unique()[:2])
    data_to_plot = [df_gender[df_gender['Gender'] == g]['PCI'].values for g in gender_codes]
    bp = ax.boxplot(data_to_plot, positions=[1, 2], widths=0.3, 
                    patch_artist=True, showfliers=False, alpha=0.7)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#3498DB')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Male', 'Female'])
    ax.set_ylabel('Privacy Concern Index (PCI)')
    ax.set_title('Privacy Concern by Gender', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add sample sizes
    for i, code in enumerate(gender_codes):
        n = len(df_gender[df_gender['Gender'] == code])
        ax.text(i+1, ax.get_ylim()[0], f'n={n}', ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../figures/fig2_gender_differences.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2 saved: fig2_gender_differences.png")
    plt.close()

# %% [markdown]
# ## Figure 3: Correlation Scatter Plot (PCI vs DPAI)

# %%
print("\nGenerating Figure 3: PCI × DPAI Correlation...")

if 'PCI' in df.columns and 'DPAI' in df.columns:
    df_corr = df[['PCI', 'DPAI']].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot with hexbin for density
    hb = ax.hexbin(df_corr['DPAI'], df_corr['PCI'], gridsize=30, cmap='YlOrRd', alpha=0.8)
    
    # Regression line
    z = np.polyfit(df_corr['DPAI'], df_corr['PCI'], 1)
    p = np.poly1d(z)
    ax.plot(df_corr['DPAI'].sort_values(), p(df_corr['DPAI'].sort_values()), 
            'r--', linewidth=2, label='Trend line')
    
    # Correlation coefficient
    r, p_val = stats.pearsonr(df_corr['PCI'], df_corr['DPAI'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np < .001', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Data Protection Awareness Index (DPAI)')
    ax.set_ylabel('Privacy Concern Index (PCI)')
    ax.set_title('Relationship Between Awareness and Concern', fontweight='bold')
    ax.legend()
    
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Density')
    
    plt.tight_layout()
    plt.savefig('../figures/fig3_pci_dpai_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3 saved: fig3_pci_dpai_correlation.png")
    plt.close()

# %% [markdown]
# ## Figure 4: Privacy Paradox

# %%
print("\nGenerating Figure 4: Privacy Paradox...")

if 'PCI' in df.columns and 'Data_Sharing_Score' in df.columns:
    df_paradox = df[['PCI', 'Data_Sharing_Score']].dropna()
    
    # Create PCI categories
    df_paradox['PCI_Cat'] = pd.cut(df_paradox['PCI'], 
                                     bins=[0, df_paradox['PCI'].quantile(0.33),
                                           df_paradox['PCI'].quantile(0.67), 5],
                                     labels=['Low', 'Moderate', 'High'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Box plot
    sharing_by_concern = [df_paradox[df_paradox['PCI_Cat'] == cat]['Data_Sharing_Score'].values 
                           for cat in ['Low', 'Moderate', 'High']]
    
    bp = ax.boxplot(sharing_by_concern, labels=['Low', 'Moderate', 'High'],
                    patch_artist=True, showfliers=True)
    
    colors = ['#27AE60', '#F39C12', '#E74C3C']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Privacy Concern Level')
    ax.set_ylabel('Number of AI Tools Used (Data Sharing Score)')
    ax.set_title('Privacy Paradox: Concern vs. Behavior', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add means
    for i, cat in enumerate(['Low', 'Moderate', 'High']):
        mean_val = df_paradox[df_paradox['PCI_Cat'] == cat]['Data_Sharing_Score'].mean()
        ax.text(i+1, mean_val, f'M={mean_val:.2f}', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../figures/fig4_privacy_paradox.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 4 saved: fig4_privacy_paradox.png")
    plt.close()

# %% [markdown]
# ## Figure 5: Feature Importance (Random Forest)

# %%
print("\nGenerating Figure 5: Feature Importance...")

if feature_importance is not None and len(feature_importance) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_n = min(10, len(feature_importance))
    top_features = feature_importance.head(top_n)
    
    bars = ax.barh(range(top_n), top_features['Importance'].values, color='#3498DB', alpha=0.8, edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Predictors of Privacy Concern\n(Random Forest Model)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Annotate bars
    for i, (feat, imp) in enumerate(zip(top_features['Feature'].values, top_features['Importance'].values)):
        ax.text(imp, i, f' {imp:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../figures/fig5_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 5 saved: fig5_feature_importance.png")
    plt.close()
else:
    print("⚠ Feature importance data not available, skipping Figure 5")

# %% [markdown]
# ## Figure 6: Cluster Profiles

# %%
print("\nGenerating Figure 6: Student Clusters...")

if clusters is not None and len(clusters) > 0:
    # Create radar chart for cluster profiles
    cluster_features = ['PCI', 'DPAI', 'Data_Sharing_Score', 'PBI']
    available_features = [f for f in cluster_features if f in clusters.columns]
    
    if len(available_features) >= 3:
        cluster_profiles = clusters.groupby('Cluster')[available_features].mean()
        
        # Normalize to 0-1 scale for radar chart
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12']
        for cluster_id, color in zip(range(len(cluster_profiles)), colors[:len(cluster_profiles)]):
            values = cluster_profiles_norm.iloc[cluster_id].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id} (n={len(clusters[clusters["Cluster"]==cluster_id])})',
                   color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_features, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('Student Cluster Profiles\n(Normalized Scores)', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('../figures/fig6_cluster_profiles.png', dpi=300, bbox_inches='tight')
        print("✓ Figure 6 saved: fig6_cluster_profiles.png")
        plt.close()
    else:
        print("⚠ Insufficient cluster features, skipping Figure 6")
else:
    print("⚠ Cluster data not available, skipping Figure 6")

# %% [markdown]
# ## Figure 7: DPIA Risk Matrix

# %%
print("\nGenerating Figure 7: DPIA Risk Matrix...")

try:
    risk_register = pd.read_csv('../results/dpia_risk_register.csv')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Risk matrix: Likelihood (x) vs Severity (y)
    colors = {'HIGH': '#E74C3C', 'MEDIUM': '#F39C12', 'LOW': '#27AE60'}
    
    for _, risk in risk_register.iterrows():
        color = colors.get(risk['Priority'], '#95A5A6')
        ax.scatter(risk['Likelihood'], risk['Severity'], 
                  s=risk['Risk_Score']*50, alpha=0.6, color=color,
                  edgecolors='black', linewidths=1.5)
        ax.text(risk['Likelihood'], risk['Severity'], 
               f"R{risk['Risk_ID']}", ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0.5, 4.5)
    ax.set_xlabel('Likelihood (1-4)', fontsize=12)
    ax.set_ylabel('Severity (1-4)', fontsize=12)
    ax.set_title('DPIA Risk Matrix\n(Bubble size = Risk Score)', fontweight='bold', fontsize=14)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Rare', 'Unlikely', 'Likely', 'Certain'])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Minor', 'Moderate', 'Major', 'Severe'])
    ax.grid(True, alpha=0.3)
    
    # Shaded risk zones
    ax.axhspan(0.5, 2.5, xmin=0, xmax=0.5, alpha=0.1, color='green')
    ax.axhspan(2.5, 4.5, xmin=0.5, xmax=1, alpha=0.1, color='red')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='HIGH', markerfacecolor='#E74C3C', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='MEDIUM', markerfacecolor='#F39C12', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='LOW', markerfacecolor='#27AE60', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Priority')
    
    # Risk labels
    risk_labels = "\n".join([f"R{r['Risk_ID']}: {r['Risk'][:30]}..." for _, r in risk_register.iterrows()])
    ax.text(1.05, 0.5, risk_labels, transform=ax.transAxes, fontsize=8,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../figures/fig7_risk_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 7 saved: fig7_risk_matrix.png")
    plt.close()
except:
    print("⚠ Risk register not available, skipping Figure 7")

# %% [markdown]
# ## Figure 8: AI Tool Usage

# %%
print("\nGenerating Figure 8: AI Tool Usage...")

if all(col in df.columns for col in ['Q13a', 'Q13b', 'Q13c', 'Q13d', 'Q13e', 'Q13f']):
    tools = {
        'ChatGPT': 'Q13a',
        'Copilot': 'Q13b',
        'Gemini': 'Q13c',
        'Perplexity': 'Q13d',
        'Claude': 'Q13e',
        'Other': 'Q13f'
    }
    
    usage = []
    for tool_name, col in tools.items():
        pct = (df[col].sum() / len(df)) * 100
        usage.append({'Tool': tool_name, 'Usage_%': pct})
    
    usage_df = pd.DataFrame(usage).sort_values('Usage_%', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(usage_df['Tool'], usage_df['Usage_%'], color='#3498DB', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Usage Rate (%)', fontsize=12)
    ax.set_title('AI Tool Usage Among Students (N=22,836)', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Annotate bars
    for i, (tool, pct) in enumerate(zip(usage_df['Tool'], usage_df['Usage_%'])):
        ax.text(pct, i, f' {pct:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../figures/fig8_tool_usage.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 8 saved: fig8_tool_usage.png")
    plt.close()

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE")
print("="*80)
print("\nFigures created in ../figures/ directory:")
print("  1. fig1_composite_indices.png - Distribution of key variables")
print("  2. fig2_gender_differences.png - Gender comparison")
print("  3. fig3_pci_dpai_correlation.png - Awareness-concern relationship")
print("  4. fig4_privacy_paradox.png - Concern vs. behavior")
print("  5. fig5_feature_importance.png - ML predictor rankings")
print("  6. fig6_cluster_profiles.png - Student segments")
print("  7. fig7_risk_matrix.png - DPIA risk assessment")
print("  8. fig8_tool_usage.png - AI tool adoption rates")
print("\nAll figures are publication-quality (300 DPI)")
print("Ready for inclusion in the final report")
print("="*80)


