import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TASK 01: [REFAC] INITIAL EDA (PURE STATIC ANALYSIS)
# ============================================================================

def load_data(path):
    print(f"\n[1] Loading raw data from {path}...")
    return pd.read_csv(path, encoding='ISO-8859-1')

def target_analysis(df):
    print("\n[2] TARGET ANALYSIS (match)")
    counts = df['match'].value_counts()
    probs = df['match'].value_counts(normalize=True)
    print("    - Value Counts:")
    print(counts)
    print("\n    - Probabilities (%):")
    print(probs * 100)
    
    if probs[1] < 0.2:
        print("\n    [INSIGHT] Class imbalance detected. Priority: F0.5-score (Precision).")

def missing_value_audit(df):
    print("\n[3] MISSING VALUE AUDIT (STATIC FEATURES)")
    # Grouping static features for profile analysis
    static_vars = [
        'age', 'gender', 'race', 'imprace', 'imprelig', 'goal', 'date', 
        'go_out', 'career_c', 'exphappy', 'expnum',
        'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
        'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
        'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1'
    ]
    
    # Check for missing values in static variables
    missing_stats = df[static_vars].isnull().mean() * 100
    print("    - Top 10 Missing Static Variables (%):")
    print(missing_stats.sort_values(ascending=False).head(10))
    
    # Entity-level audit (Ghost Users)
    # Each user (iid) should have a unique profile
    profiles = df.groupby('iid')[static_vars].first()
    missing_per_user = profiles.isnull().sum(axis=1)
    ghosts = missing_per_user[missing_per_user >= 20].index.tolist()
    
    print(f"\n[4] ENTITY INTEGRITY AUDIT")
    print(f"    - Identified {len(ghosts)} 'ghost' users missing >= 20/27 core static features.")
    print(f"    - Ghost iids: {ghosts}")
    print(f"    - These users and their {len(df[df['iid'].isin(ghosts)])} interactions will be dropped in GĐ 3.")

def basic_stats(df):
    print("\n[5] BASIC DESCRIPTIVE STATS")
    print(f"    - Age distribution: Mean={df['age'].mean():.2f}, Median={df['age'].median():.2f}")
    print(f"    - Gender distribution: {df['gender'].value_counts(normalize=True).to_dict()}")

if __name__ == "__main__":
    DATA_PATH = 'Data/Speed Dating Data.csv'
    df = load_data(DATA_PATH)
    
    print("=" * 80)
    print("TASK 01: FRAMING & INITIAL EDA (ANTI-LEAKAGE PHILOSOPHY)")
    print("=" * 80)
    
    target_analysis(df)
    missing_value_audit(df)
    basic_stats(df)
    
    print("\n" + "=" * 80)
    print("✓ TASK 01 COMPLETED: STATIC FEATURE BASELINE ESTABLISHED.")
    print("=" * 80)
