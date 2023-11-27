import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load your data into a pandas DataFrame (assuming you have columns: 'letter_text', 'label')
# 'letter_text' contains the referral letter text, 'label' contains the surgery outcome (0 or 1)
# 0: No surgery, 1: Surgery

# Sample data
data = {
    'letter_text': ['Patient was referred for surgery', 'No surgery required', 'Surgery is recommended'],
    'label': [1, 0, 1]
}
df = pd.DataFrame(data)

# Define a list of words or phrases you want to analyze
keywords = ['surgery', 'recommended', 'referral']

# Create a new column for each keyword to indicate presence or absence
for keyword in keywords:
    df[keyword] = df['letter_text'].str.contains(keyword, case=False)

# Create a contingency table for each keyword
contingency_tables = {}
for keyword in keywords:
    contingency_table = pd.crosstab(df[keyword], df['label'])
    contingency_tables[keyword] = contingency_table

# Perform Chi-Square test for each keyword
results = {}
for keyword, table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(table)
    results[keyword] = {'chi2': chi2, 'p_value': p}

# Print results
for keyword, result in results.items():
    print(f"Keyword: {keyword}")
    print(f"Chi-Square Value: {result['chi2']:.2f}")
    print(f"P-Value: {result['p_value']:.4f}")
    print("Association: Significant" if result['p_value'] < 0.05 else "Association: Not Significant")
    print("="*50)
