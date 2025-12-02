#!/usr/bin/env python3
"""
DPIA Risk Assessment
Privacy, Security, and Compliance of GenAI in LMS: DPIA Study

Module: 7150CEM MSc Data Science Project
Date: November 2025
Author: Subhash Yadav

Creates evidence-based DPIA risk scoring using survey data.
Risk Score = Likelihood × Severity
"""

# %% [markdown]
# # Data Protection Impact Assessment (DPIA)
# Empirical risk assessment based on student survey data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.random.seed(42)
sns.set_style('whitegrid')

print("✓ Libraries loaded")

# %% [markdown]
# ## 1. Load Dataset

# %%
df = pd.read_csv('../data_proc/survey_with_indices.csv')
print(f"Dataset: {df.shape[0]:,} students")

# %% [markdown]
# ## 2. Define DPIA Risk Framework
# 
# UK GDPR requires assessment of:
# - Likelihood (1-4): How probable is the risk?
# - Severity (1-4): How serious would the impact be?
# - Risk Score = Likelihood × Severity (1-16)
#
# Classification:
# - HIGH: Score ≥ 9 (immediate action required)
# - MEDIUM: Score 5-8 (mitigation needed)
# - LOW: Score ≤ 4 (monitor)

# %%
print("\n" + "="*80)
print("DPIA RISK ASSESSMENT FRAMEWORK")
print("="*80)

risks = {
    1: {
        'name': 'Personal Data Exposure',
        'description': 'Students share personal information with GenAI systems',
        'data_source': 'Q13a-f (tool usage), PCI',
        'likelihood_basis': 'Proportion using AI tools',
        'severity_basis': 'Mean concern about privacy invasion (Q22f)'
    },
    2: {
        'name': 'Data Breach Risk',
        'description': 'Third-party AI providers could be breached',
        'data_source': 'Data_Sharing_Score',
        'likelihood_basis': 'Number of tools used (more tools = more exposure)',
        'severity_basis': 'Student concern about data breaches'
    },
    3: {
        'name': 'Unauthorized Data Profiling',
        'description': 'AI systems profile students without explicit consent',
        'data_source': 'Policy_Aware (Q20)',
        'likelihood_basis': 'Lack of policy awareness',
        'severity_basis': 'Concern about misleading info (Q22e)'
    },
    4: {
        'name': 'Academic Integrity Violation',
        'description': 'Encourages unethical behavior or plagiarism',
        'data_source': 'Q22b, Q22c',
        'likelihood_basis': 'High usage rates',
        'severity_basis': 'Concern about cheating/plagiarism'
    },
    5: {
        'name': 'Lack of Transparency',
        'description': 'Students unaware of data processing',
        'data_source': 'DPAI, Policy_Aware',
        'likelihood_basis': 'Low awareness scores',
        'severity_basis': 'Regulatory requirement (UK GDPR Article 13)'
    },
    6: {
        'name': 'Cross-Border Data Transfer',
        'description': 'Student data transferred outside UK/EEA without safeguards',
        'data_source': 'Q13a-f (tool usage)',
        'likelihood_basis': 'Use of non-EU providers',
        'severity_basis': 'High (GDPR Chapter V violation risk)'
    }
}

print(f"\n{len(risks)} key privacy risks identified")

# %% [markdown]
# ## 3. Calculate Risk Scores

# %%
print("\n" + "="*80)
print("RISK SCORING")
print("="*80)

risk_scores = []

# Risk 1: Personal Data Exposure
if 'Data_Sharing_Score' in df.columns and 'PCI' in df.columns:
    sharing_rate = (df['Data_Sharing_Score'] > 0).mean()
    likelihood_1 = int(np.ceil(sharing_rate * 4))  # Convert to 1-4 scale
    
    # Severity: mean concern about privacy invasion
    if 'Q22f' in df.columns:
        severity_1 = int(np.ceil(df['Q22f'].mean() * 4 / 5))  # Convert 1-5 to 1-4
    else:
        severity_1 = 3  # Default: moderate-high
    
    risk_score_1 = likelihood_1 * severity_1
    risk_scores.append({
        'Risk_ID': 1,
        'Risk': risks[1]['name'],
        'Likelihood': likelihood_1,
        'Likelihood_Evidence': f'{sharing_rate*100:.1f}% use AI tools',
        'Severity': severity_1,
        'Severity_Evidence': 'Privacy invasion concern scores',
        'Risk_Score': risk_score_1,
        'Priority': 'HIGH' if risk_score_1 >= 9 else ('MEDIUM' if risk_score_1 >= 5 else 'LOW')
    })
    
    print(f"\nRisk 1: {risks[1]['name']}")
    print(f"  Likelihood: {likelihood_1}/4 ({sharing_rate*100:.1f}% use tools)")
    print(f"  Severity: {severity_1}/4")
    print(f"  Risk Score: {risk_score_1}/16 [{risk_scores[-1]['Priority']}]")

# Risk 2: Data Breach Risk
if 'Data_Sharing_Score' in df.columns:
    mean_tools = df['Data_Sharing_Score'].mean()
    likelihood_2 = int(np.ceil(mean_tools / 1.5))  # Scale to 1-4
    likelihood_2 = min(likelihood_2, 4)
    severity_2 = 4  # High severity (potential for large-scale breach)
    
    risk_score_2 = likelihood_2 * severity_2
    risk_scores.append({
        'Risk_ID': 2,
        'Risk': risks[2]['name'],
        'Likelihood': likelihood_2,
        'Likelihood_Evidence': f'Mean {mean_tools:.1f} tools used',
        'Severity': severity_2,
        'Severity_Evidence': 'Third-party breach = high impact',
        'Risk_Score': risk_score_2,
        'Priority': 'HIGH' if risk_score_2 >= 9 else ('MEDIUM' if risk_score_2 >= 5 else 'LOW')
    })
    
    print(f"\nRisk 2: {risks[2]['name']}")
    print(f"  Likelihood: {likelihood_2}/4")
    print(f"  Severity: {severity_2}/4")
    print(f"  Risk Score: {risk_score_2}/16 [{risk_scores[-1]['Priority']}]")

# Risk 3: Unauthorized Profiling
if 'Policy_Aware' in df.columns:
    unaware_rate = (df['Policy_Aware'] != 1).mean()  # Assuming 1 = Yes
    likelihood_3 = int(np.ceil(unaware_rate * 4))
    severity_3 = 3  # Moderate-high
    
    risk_score_3 = likelihood_3 * severity_3
    risk_scores.append({
        'Risk_ID': 3,
        'Risk': risks[3]['name'],
        'Likelihood': likelihood_3,
        'Likelihood_Evidence': f'{unaware_rate*100:.1f}% unaware of policies',
        'Severity': severity_3,
        'Severity_Evidence': 'Profiling without consent',
        'Risk_Score': risk_score_3,
        'Priority': 'HIGH' if risk_score_3 >= 9 else ('MEDIUM' if risk_score_3 >= 5 else 'LOW')
    })
    
    print(f"\nRisk 3: {risks[3]['name']}")
    print(f"  Likelihood: {likelihood_3}/4 ({unaware_rate*100:.1f}% unaware)")
    print(f"  Severity: {severity_3}/4")
    print(f"  Risk Score: {risk_score_3}/16 [{risk_scores[-1]['Priority']}]")

# Risk 4: Academic Integrity
if 'Q22b' in df.columns and 'Q22c' in df.columns:
    cheating_concern = df['Q22b'].mean()
    plagiarism_concern = df['Q22c'].mean()
    avg_concern = (cheating_concern + plagiarism_concern) / 2
    
    likelihood_4 = 3  # High usage rates
    severity_4 = int(np.ceil(avg_concern * 4 / 5))
    
    risk_score_4 = likelihood_4 * severity_4
    risk_scores.append({
        'Risk_ID': 4,
        'Risk': risks[4]['name'],
        'Likelihood': likelihood_4,
        'Likelihood_Evidence': 'High usage in academic work',
        'Severity': severity_4,
        'Severity_Evidence': f'Mean concern: {avg_concern:.2f}/5',
        'Risk_Score': risk_score_4,
        'Priority': 'HIGH' if risk_score_4 >= 9 else ('MEDIUM' if risk_score_4 >= 5 else 'LOW')
    })
    
    print(f"\nRisk 4: {risks[4]['name']}")
    print(f"  Likelihood: {likelihood_4}/4")
    print(f"  Severity: {severity_4}/4")
    print(f"  Risk Score: {risk_score_4}/16 [{risk_scores[-1]['Priority']}]")

# Risk 5: Lack of Transparency
if 'DPAI' in df.columns:
    low_awareness_rate = (df['DPAI'] < 3).mean()  # Below midpoint
    likelihood_5 = int(np.ceil(low_awareness_rate * 4))
    severity_5 = 4  # High (regulatory requirement)
    
    risk_score_5 = likelihood_5 * severity_5
    risk_scores.append({
        'Risk_ID': 5,
        'Risk': risks[5]['name'],
        'Likelihood': likelihood_5,
        'Likelihood_Evidence': f'{low_awareness_rate*100:.1f}% low awareness',
        'Severity': severity_5,
        'Severity_Evidence': 'GDPR transparency requirement',
        'Risk_Score': risk_score_5,
        'Priority': 'HIGH' if risk_score_5 >= 9 else ('MEDIUM' if risk_score_5 >= 5 else 'LOW')
    })
    
    print(f"\nRisk 5: {risks[5]['name']}")
    print(f"  Likelihood: {likelihood_5}/4")
    print(f"  Severity: {severity_5}/4")
    print(f"  Risk Score: {risk_score_5}/16 [{risk_scores[-1]['Priority']}]")

# Risk 6: Cross-Border Transfer
if 'Q13a' in df.columns:  # ChatGPT usage (US-based)
    intl_transfer_rate = df['Q13a'].mean()
    likelihood_6 = int(np.ceil(intl_transfer_rate * 4))
    severity_6 = 4  # High (GDPR Chapter V)
    
    risk_score_6 = likelihood_6 * severity_6
    risk_scores.append({
        'Risk_ID': 6,
        'Risk': risks[6]['name'],
        'Likelihood': likelihood_6,
        'Likelihood_Evidence': f'{intl_transfer_rate*100:.1f}% use non-EU tools',
        'Severity': severity_6,
        'Severity_Evidence': 'GDPR Chapter V violation risk',
        'Risk_Score': risk_score_6,
        'Priority': 'HIGH' if risk_score_6 >= 9 else ('MEDIUM' if risk_score_6 >= 5 else 'LOW')
    })
    
    print(f"\nRisk 6: {risks[6]['name']}")
    print(f"  Likelihood: {likelihood_6}/4")
    print(f"  Severity: {severity_6}/4")
    print(f"  Risk Score: {risk_score_6}/16 [{risk_scores[-1]['Priority']}]")

# %% [markdown]
# ## 4. Risk Register

# %%
risk_register = pd.DataFrame(risk_scores)
risk_register = risk_register.sort_values('Risk_Score', ascending=False)

print("\n" + "="*80)
print("DPIA RISK REGISTER (Ranked by Risk Score)")
print("="*80)
print("\n" + risk_register.to_string(index=False))

# Save risk register
import os
os.makedirs('../results', exist_ok=True)
risk_register.to_csv('../results/dpia_risk_register.csv', index=False)
print("\n✓ Risk register saved to results/dpia_risk_register.csv")

# %% [markdown]
# ## 5. Mitigation Strategies

# %%
print("\n" + "="*80)
print("MITIGATION STRATEGIES FOR HIGH-PRIORITY RISKS")
print("="*80)

high_risks = risk_register[risk_register['Priority'] == 'HIGH']

if len(high_risks) > 0:
    print(f"\n{len(high_risks)} HIGH priority risks require immediate action:\n")
    
    for _, risk in high_risks.iterrows():
        print(f"\n{'─'*80}")
        print(f"Risk {risk['Risk_ID']}: {risk['Risk']} [Score: {risk['Risk_Score']}/16]")
        print(f"{'─'*80}")
        
        # Mitigation recommendations
        if risk['Risk_ID'] == 1:
            print("Mitigation Strategies:")
            print("  1. Implement privacy-by-design in LMS-AI integrations")
            print("  2. Minimize data collection to necessary categories only")
            print("  3. Provide clear opt-out mechanisms for students")
            print("  4. Regular privacy audits of AI tool usage")
            
        elif risk['Risk_ID'] == 2:
            print("Mitigation Strategies:")
            print("  1. Vendor due diligence: assess security certifications")
            print("  2. Data Processing Agreements with all AI providers")
            print("  3. Encryption in transit and at rest")
            print("  4. Incident response plan for third-party breaches")
            
        elif risk['Risk_ID'] == 3:
            print("Mitigation Strategies:")
            print("  1. Clear AI usage policies communicated to all students")
            print("  2. Explicit consent mechanisms before data processing")
            print("  3. Right to object to automated decision-making")
            print("  4. Transparency about profiling purposes")
            
        elif risk['Risk_ID'] == 4:
            print("Mitigation Strategies:")
            print("  1. Academic integrity guidelines updated for GenAI")
            print("  2. Detection tools for AI-generated content")
            print("  3. Assessment redesign to reduce cheating opportunities")
            print("  4. Student training on ethical AI use")
            
        elif risk['Risk_ID'] == 5:
            print("Mitigation Strategies:")
            print("  1. Comprehensive privacy notices (GDPR Article 13)")
            print("  2. Layered privacy information (short + detailed)")
            print("  3. Student awareness campaigns")
            print("  4. Easy access to data protection policies")
            
        elif risk['Risk_ID'] == 6:
            print("Mitigation Strategies:")
            print("  1. Conduct Transfer Impact Assessments (TIAs)")
            print("  2. Use Standard Contractual Clauses (SCCs)")
            print("  3. Prefer EU-based AI providers where possible")
            print("  4. Supplementary measures for US transfers")
else:
    print("\n✓ No HIGH priority risks identified")

# %% [markdown]
# ## 6. DPIA Summary

# %%
print("\n" + "="*80)
print("DPIA SUMMARY")
print("="*80)

total_risks = len(risk_register)
high_risks_count = len(risk_register[risk_register['Priority'] == 'HIGH'])
medium_risks_count = len(risk_register[risk_register['Priority'] == 'MEDIUM'])
low_risks_count = len(risk_register[risk_register['Priority'] == 'LOW'])

print(f"\nTotal Risks Assessed: {total_risks}")
print(f"  • HIGH Priority: {high_risks_count} (immediate action required)")
print(f"  • MEDIUM Priority: {medium_risks_count} (mitigation needed)")
print(f"  • LOW Priority: {low_risks_count} (monitor)")

print(f"\nHighest Risk: {risk_register.iloc[0]['Risk']}")
print(f"  Risk Score: {risk_register.iloc[0]['Risk_Score']}/16")

print("\n### KEY RECOMMENDATIONS ###")
print("\n1. IMMEDIATE ACTIONS:")
print("   • Complete vendor due diligence for all AI providers")
print("   • Establish clear AI usage policies")
print("   • Implement privacy notices compliant with UK GDPR")

print("\n2. SHORT-TERM (1-3 months):")
print("   • Student awareness campaigns on data protection")
print("   • Academic integrity guidelines update")
print("   • Privacy-by-design review of LMS integrations")

print("\n3. ONGOING:")
print("   • Regular DPIA reviews (annually or when systems change)")
print("   • Monitor ICO guidance on AI in education")
print("   • Student feedback mechanisms on privacy concerns")

# Create summary document
dpia_summary = {
    'Assessment_Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'Total_Risks': total_risks,
    'HIGH_Priority': high_risks_count,
    'MEDIUM_Priority': medium_risks_count,
    'LOW_Priority': low_risks_count,
    'Highest_Risk': risk_register.iloc[0]['Risk'],
    'Highest_Risk_Score': risk_register.iloc[0]['Risk_Score'],
    'Data_Source': f"{df.shape[0]:,} student survey responses",
    'Compliance_Framework': 'UK GDPR (Data Protection Act 2018)'
}

dpia_summary_df = pd.DataFrame([dpia_summary])
dpia_summary_df.to_csv('../results/dpia_summary.csv', index=False)

print("\n" + "="*80)
print("DPIA ASSESSMENT COMPLETE")
print("="*80)
print("\n✓ Risk register created")
print("✓ Mitigation strategies documented")
print("✓ DPIA summary generated")
print("\nFiles saved:")
print("  • results/dpia_risk_register.csv")
print("  • results/dpia_summary.csv")
print("\nThis DPIA provides evidence-based risk assessment for")
print("GenAI integrations in UK higher education LMS platforms.")
print("="*80)


