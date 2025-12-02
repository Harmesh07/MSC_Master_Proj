#!/usr/bin/env python3
"""
Data Acquisition Script
Privacy, Security, and Compliance of GenAI in LMS: DPIA Study

This script performs initial data acquisition and inspection.
Convert to Jupyter notebook using: jupyter nbconvert --to notebook 01_data_acquisition.py
"""

# %% [markdown]
# # Data Acquisition Notebook
# ## Privacy, Security, and Compliance of GenAI in LMS: DPIA Study

# %% [markdown]
# ## 1. Environment Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
np.random.seed(42)
sns.set_style('whitegrid')

print(f"Environment setup complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ## 2. Data Loading

# %%
RAW_DATA_PATH = '../data_raw/final dataset.xlsx'
PROCESSED_DATA_PATH = '../data_proc/'

print("Loading raw dataset...")
df_raw = pd.read_excel(RAW_DATA_PATH)
print(f"✓ Dataset loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

# %% [markdown]
# ## 3. Initial Inspection

# %%
print("=== FIRST 5 ROWS ===")
print(df_raw.head())

print("\n=== DATASET INFO ===")
print(df_raw.info())

# %% [markdown]
# ## 4. Missing Data Assessment

# %%
missing_summary = pd.DataFrame({
    'Variable': df_raw.columns,
    'Missing_Count': df_raw.isnull().sum(),
    'Missing_Percent': (df_raw.isnull().sum() / len(df_raw)) * 100
}).sort_values('Missing_Percent', ascending=False)

print("=== MISSING DATA SUMMARY ===")
print(missing_summary.head(20))

# %% [markdown]
# ## 5. Save Metadata

# %%
missing_summary.to_csv(f'{PROCESSED_DATA_PATH}metadata_summary.csv', index=False)
df_raw.to_csv(f'{PROCESSED_DATA_PATH}survey_raw_copy.csv', index=False)
print("✓ Files saved to data_proc/")
