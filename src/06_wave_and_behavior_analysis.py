import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TASK 06: WAVE & BEHAVIOR ANALYSIS (Kaggle-Inspired: Breaking the Waves)
# ============================================================================
CONFIG = {
    'paths': {
        'raw_data': 'Data/Speed Dating Data.csv',
        'final_data': 'Data/data_final_v2.csv',
        'analysis_dir': 'Logs/'
    }
}

print("=" * 80)
print("TASK 06: WAVE & BEHAVIOR ANALYSIS (Understanding Experimental Variations)")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[0] Loading data...")
df_raw = pd.read_csv(CONFIG['paths']['raw_data'], encoding='ISO-8859-1')
df_final = pd.read_csv(CONFIG['paths']['final_data'])

# ============================================================================
# ANALYSIS 1: Wave Impact on Decision & Match Rate
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: WAVE IMPACT ON DECISIONS & MATCH RATES")
print("="*80)

wave_analysis = df_raw.groupby('wave').agg({
    'match': ['count', 'sum', 'mean'],  # Total, Matches, Match Rate
    'dec': ['mean'],                     # Yes Rate
    'wave': ['min']
}).round(4)

wave_analysis.columns = ['Total_Dates', 'Total_Matches', 'Match_Rate', 'Yes_Rate']
wave_analysis['Yes_Rate'] = wave_analysis['Yes_Rate'].astype(float)
wave_analysis = wave_analysis[['Total_Dates', 'Total_Matches', 'Match_Rate', 'Yes_Rate']]

print("\nWave Statistics:")
print(wave_analysis)

# Save analysis
wave_analysis.to_csv(CONFIG['paths']['analysis_dir'] + '06_wave_impact.csv')

# ============================================================================
# ANALYSIS 2: Demographic Representation by Wave
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 2: GENDER & RACE DISTRIBUTION BY WAVE")
print("="*80)

# Unique participants per wave
unique_per_wave = df_raw.groupby('wave')['iid'].nunique()
print(f"\nUnique participants per wave:")
print(unique_per_wave)

# Gender split by wave
gender_by_wave = df_raw[['wave', 'gender']].drop_duplicates().groupby(['wave', 'gender']).size().unstack(fill_value=0)
gender_by_wave.columns = ['Female', 'Male']
print(f"\nGender distribution by wave (unique users):")
print(gender_by_wave)

# ============================================================================
# ANALYSIS 3: Race & Expectation Effects by Wave
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 3: RACE EFFECT & EXPECTATION MISMATCH BY WAVE")
print("="*80)

# Same race effect by wave
samerace_effect = df_raw.groupby(['wave', 'samerace']).agg({
    'dec': 'mean',
    'match': 'mean'
}).round(4)

print("\nYes rate by Same Race status per wave:")
for wave in sorted(df_raw['wave'].unique()):
    mask_same = (df_raw['wave'] == wave) & (df_raw['samerace'] == 1)
    mask_diff = (df_raw['wave'] == wave) & (df_raw['samerace'] == 0)
    
    if mask_same.any() and mask_diff.any():
        yes_same = df_raw[mask_same]['dec'].mean()
        yes_diff = df_raw[mask_diff]['dec'].mean()
        effect = yes_same - yes_diff
        print(f"  Wave {int(wave):2d}: Same Race={yes_same:.3f} | Diff Race={yes_diff:.3f} | Effect={effect:+.3f}")

# ============================================================================
# ANALYSIS 4: Preference Gaps (Declared vs Actual)
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 4: EXPECTATION VS REALITY (Kaggle Insight)")
print("="*80)

print("\nExpectation Analysis by Wave:")
print("(Before date: exphappy, After date: match decision)")

expect_reality = df_raw.groupby('wave').agg({
    'exphappy': 'mean',
    'dec': 'mean',
    'match': 'mean'
}).round(4)

expect_reality.columns = ['Avg_Expected_Happiness', 'Actual_Yes_Rate', 'Actual_Match_Rate']

print(expect_reality)

# Expectation-Reality gap
expect_reality['Happiness_vs_Yes_Gap'] = expect_reality['Avg_Expected_Happiness'] - expect_reality['Actual_Yes_Rate']
print("\nGap Analysis (Expected Happiness vs Actual Yes Rate):")
print(expect_reality[['Avg_Expected_Happiness', 'Actual_Yes_Rate', 'Happiness_vs_Yes_Gap']])

expect_reality.to_csv(CONFIG['paths']['analysis_dir'] + '06_expectation_vs_reality.csv')

# ============================================================================
# ANALYSIS 5: Feature Stability Across Waves
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 5: FEATURE STABILITY & CONSISTENCY ACROSS WAVES")
print("="*80)

# Check which waves have data for key survey questions
survey_cols = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']

feature_coverage = df_raw.groupby('wave')[survey_cols].apply(
    lambda x: (x.notna().sum() / len(x)).round(3)
)

print("\nData completeness for key survey questions by wave:")
print(feature_coverage)

# Identify waves with different scale ranges
print("\nWave Scale Characteristics (are there 1-10 vs 0-100 scales?):")
for col in survey_cols[:2]:  # Check first 2 columns
    print(f"\n{col}:")
    for wave in sorted(df_raw['wave'].unique()):
        mask = df_raw['wave'] == wave
        valid_data = df_raw[mask][col].dropna()
        if len(valid_data) > 0:
            print(f"  Wave {int(wave):2d}: Range=[{valid_data.min():.0f}-{valid_data.max():.0f}], N={len(valid_data)}")

# ============================================================================
# ANALYSIS 6: Model Performance Variation by Wave
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 6: INSIGHTS FOR WAVE-AWARE MODELING")
print("="*80)

print("""
Key Findings to Consider:

1. WAVE EFFECT: If match rates vary significantly across waves, the model needs
   to account for different experimental conditions. Consider:
   - Adding wave as a feature (categorical encoding)
   - Wave-specific hyperparameter tuning
   - Separate models for waves with different characteristics

2. SCALE NORMALIZATION: If waves use different rating scales (1-10 vs 0-100),
   ensure normalization is applied correctly (already handled in Task 03)

3. SAMPLE IMBALANCE: Waves with very few matches may cause training issues.
   Monitor class imbalance per wave.

4. EXPECTATION-REALITY GAP: If expected happiness >> actual yes rate, 
   participants may be optimistic. Tree models should capture this pattern.

5. FAIRNESS ACROSS WAVES: Check if the model's performance degrades for 
   certain waves (important for robustness).
""")

# Save summary
summary_data = {
    'Analysis': [
        'Wave-specific match rates',
        'Race effects by wave',
        'Expectation-reality gaps',
        'Feature stability',
        'Sample sizes'
    ],
    'Output_File': [
        'fairness_by_wave.csv (from Task 05)',
        'Part of 06_wave_impact.csv',
        '06_expectation_vs_reality.csv',
        'Check survey column coverage',
        '06_wave_impact.csv'
    ]
}

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
for f in ['06_wave_impact.csv', '06_expectation_vs_reality.csv']:
    print(f"  - {CONFIG['paths']['analysis_dir']}{f}")

print("\nRecommendations:")
print("  1. Review wave-specific match rates for potential data quality issues")
print("  2. Monitor fairness metrics by wave (use Task 05 output)")
print("  3. Consider wave as a feature in Task 04 if significant variation detected")
print("  4. Validate model generalization across all waves (see fairness analysis)")
