import pandas as pd
import numpy as np
import os

# Create plots directory if not exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load data
df = pd.read_csv('Data/Speed Dating Data.csv', encoding='ISO-8859-1')

print("--- Step 1: Choice Overload (H1) ---")
# condtn: 1 (Limited choice), 2 (Many choices)
choice_overload = df.groupby('condtn')['match'].mean()
print(choice_overload)

print("\n--- Step 2: Correlation Analysis (H2 - Data Leakage) ---")
static_vars = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
dynamic_vars = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
target = ['match']

# Calculate correlation with target
static_corr = df[static_vars + target].corr()['match'].sort_values(ascending=False)
dynamic_corr = df[dynamic_vars + target].corr()['match'].sort_values(ascending=False)

print("Correlation (Static Variables):")
print(static_corr)
print("\nCorrelation (Dynamic Variables):")
print(dynamic_corr)

print("\n--- Step 3: Age Gap Analysis (H3) ---")
# Need to handle missing values for age_o and age
df['age_diff'] = abs(df['age'] - df['age_o'])
age_analysis = df.groupby('match')['age_diff'].describe()
print(age_analysis)

