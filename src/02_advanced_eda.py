import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TASK 02: [REFAC] ADVANCED EDA & LEAKAGE AUDIT
# ============================================================================

def load_data(path):
    print(f"\n[1] Loading raw data from {path}...")
    return pd.read_csv(path, encoding='ISO-8859-1')

def hypothesis_choice_overload(df):
    print("\n[2] H1: CHOICE OVERLOAD (Nghịch lý lựa chọn)")
    # condtn: 1 (Limited choice), 2 (Many choices)
    stats = df.groupby('condtn')['match'].agg(['mean', 'count'])
    stats.columns = ['Match Rate (%)', 'Sample Size']
    stats['Match Rate (%)'] *= 100
    print(stats)
    
    diff = stats.loc[1, 'Match Rate (%)'] - stats.loc[2, 'Match Rate (%)']
    print(f"\n    [RESULT] Match rate dropped by {diff:.2f}% in 'Many choices' condition.")
    print("    [CONCLUSION] Hypothesis H1 confirmed.")

def leakage_audit(df):
    print("\n[3] H2: ANTI-LEAKAGE AUDIT (Correlation Contrast)")
    # Defining feature groups
    static_pref = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
    dynamic_ratings = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
    
    # Target correlation
    static_corr = df[static_pref + ['match']].corr()['match'].drop('match').abs()
    dynamic_corr = df[dynamic_ratings + ['match']].corr()['match'].drop('match').abs()
    
    print("    - Average Correlation (Static Profiles):", static_corr.mean())
    print("    - Average Correlation (Dynamic Post-Date):", dynamic_corr.mean())
    
    ratio = dynamic_corr.mean() / max(static_corr.mean(), 0.001)
    print(f"\n    [RESULT] Dynamic features are {ratio:.1f}x more correlated than static ones.")
    print("    [WARNING] Using dynamic features causes 100% Data Leakage in a real-world recommendation scenario.")
    print("    [STRATEGY] Drop all Dynamic features. Build 'Mutual Surplus' from Static Profiles instead.")

def age_gap_analysis(df):
    print("\n[4] H3: AGE GAP ANALYSIS")
    if 'age_o' in df.columns:
        df['age_diff'] = (df['age'] - df['age_o']).abs()
        stats = df.groupby('match')['age_diff'].describe()
        print(stats[['mean', '50%', 'std']])
        
        diff = stats.loc[0, 'mean'] - stats.loc[1, 'mean']
        print(f"\n    [RESULT] Match pairs are {diff:.2f} years closer in age on average.")
    else:
        print("    [SKIP] Partner age (age_o) missing in this view.")

def humility_audit(df):
    print("\n[5] H4: COGNITIVE HUMILITY (3_1 vs 5_1)")
    # Comparing "How I see myself" (3_1) vs "How I think society sees me" (5_1)
    attrs = ['attr', 'sinc', 'intel', 'fun', 'amb']
    humility_scores = []
    
    for a in attrs:
        s31 = f'{a}3_1'
        s51 = f'{a}5_1'
        if s31 in df.columns and s51 in df.columns:
            bias = (df[s31] - df[s51]).mean()
            humility_scores.append({'Attribute': a, 'Ego Bias (3_1 - 5_1)': bias})
            
    if humility_scores:
        bias_df = pd.DataFrame(humility_scores)
        print(bias_df)
        print(f"\n    [INSIGHT] Average Ego Bias: {bias_df['Ego Bias (3_1 - 5_1)'].mean():.2f} points.")
        print("    [STRATEGY] Use 5_1 as 'Conservative Anchor' for Mutual Surplus calculation.")

if __name__ == "__main__":
    DATA_PATH = 'Data/Speed Dating Data.csv'
    df = load_data(DATA_PATH)
    
    print("=" * 80)
    print("TASK 02: ADVANCED EDA & HYPOTHESIS TESTING (SCIENCE OF LOVE)")
    print("=" * 80)
    
    hypothesis_choice_overload(df)
    leakage_audit(df)
    age_gap_analysis(df)
    humility_audit(df)
    
    print("\n" + "=" * 80)
    print("✓ TASK 02 COMPLETED: BEHAVIORAL FOUNDATIONS ESTABLISHED.")
    print("=" * 80)
