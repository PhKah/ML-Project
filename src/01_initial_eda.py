import pandas as pd
import numpy as np

# Step 1: Load data
try:
    df = pd.read_csv('Data/Speed Dating Data.csv', encoding='ISO-8859-1')
    print(f"--- Step 1: Shape ---")
    print(f"Kích thước: {df.shape}")
    print("\n--- df.info() summary ---")
    # Capturing basic info without printing full list to avoid context overflow
    print(f"Số lượng cột: {len(df.columns)}")
    print(f"Số lượng bản ghi: {len(df)}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Step 2: Target Analysis
print("\n--- Step 2: Target Analysis (match) ---")
counts = df['match'].value_counts()
probs = df['match'].value_counts(normalize=True)
print("Số lượng:")
print(counts)
print("\nTỷ lệ %:")
print(probs * 100)

# Step 3: Essential variables missing stats
# Note: Added 'dec' as requested in bare_minimum as a supplementary variable
essential_vars = ['gender', 'age', 'race', 'imprace', 'imprelig', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'condtn', 'match']
print("\n--- Step 3: Missing Stats (%) ---")
missing_stats = df[essential_vars].isnull().mean() * 100
print(missing_stats.sort_values(ascending=False))

# Step 4: Diversity check
print("\n--- Step 4: Unique Values ---")
print(df[essential_vars].nunique())

# Step 5: Age Range Check
print("\n--- Step 5: Age describe ---")
print(df['age'].describe())

