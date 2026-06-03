"""
Task 03: Advanced Data Preparation (Chuẩn bị & Tiền xử lý dữ liệu)
=================================================================
Áp dụng 6 nguyên lý tiền xử lý từ Specs/plan.md:
1. Entity vs Relationship Cleaning
2. Order-Dependent Feature Engineering (DAG)
3. Scaling Strategy (MinMax vs Standard)
4. Meaningful Aggregation (17 hobbies -> 5 features)
5. Outlier Handling (IQR clip > drop)
6. Data Synchronization (referential integrity)

Author: Data Science Project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TASK 03: ADVANCED DATA PREPARATION")
print("=" * 80)

# ============================================================================
# LOAD RAW DATA
# ============================================================================
print("\n[1] Loading raw data...")
df = pd.read_csv('Data/Speed Dating Data.csv', encoding='ISO-8859-1')
print(f"   Raw data shape: {df.shape}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# ============================================================================
# STEP A: ENTITY-LEVEL CLEANING (User Level Data)
# ============================================================================
print("\n" + "=" * 80)
print("STEP A: ENTITY-LEVEL CLEANING (User Data)")
print("=" * 80)

# Define entity columns (user-level: age, gender, race, hobbies)
entity_cols = ['age', 'gender', 'race', 'sports', 'tvsports', 
               'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 
               'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 
               'music', 'shopping', 'yoga']

# A1: Identify users with too many missing values (>= 5)
print("\n[A1] Removing users with >= 5 missing entity values...")
entity_missing = df[entity_cols].isnull().sum(axis=1)
users_with_excess_missing = df[entity_missing >= 5].index
dropped_iids = df.loc[users_with_excess_missing, 'iid'].unique()
print(f"   Found {len(dropped_iids)} users with >= 5 missing values")
print(f"   Dropped iids: {sorted(dropped_iids)}")

df_clean = df[~df['iid'].isin(dropped_iids)].copy()
print(f"   Dataset after removal: {df_clean.shape}")

# A2: Save raw age BEFORE scaling (needed for age_gap calculation)
print("\n[A2] Saving raw age values for age_gap calculation...")
age_map = df_clean[['iid', 'age']].drop_duplicates().set_index('iid')['age'].to_dict()
print(f"   Saved age_map with {len(age_map)} unique users")

# A3: Impute missing entity values (Median for numeric, Mode for categorical)
print("\n[A3] Imputing missing entity values...")
for col in entity_cols:
    if col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            median_val = df_clean[col].median()
            n_missing = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(median_val)  # Use assignment
            if n_missing > 0:
                print(f"   {col}: imputed {n_missing} with median {median_val:.2f}")
        else:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else df_clean[col].dropna().iloc[0]
            n_missing = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(mode_val)  # Use assignment
            if n_missing > 0:
                print(f"   {col}: imputed {n_missing} with mode")

# A4: Apply IQR Clip for outlier handling
print("\n[A4] Applying IQR clipping for numerical entity columns...")
numeric_entity_cols = [c for c in entity_cols if c in df_clean.columns and df_clean[c].dtype in ['int64', 'float64']]
for col in numeric_entity_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    n_clipped_lower = (df_clean[col] < lower_bound).sum()
    n_clipped_upper = (df_clean[col] > upper_bound).sum()
    
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    if n_clipped_lower + n_clipped_upper > 0:
        print(f"   {col}: clipped {n_clipped_lower} lower + {n_clipped_upper} upper values")

# Save entity-cleaned dataset
df_clean.to_csv('Data/data_cleaned_entities.csv', index=False, encoding='utf-8')
print(f"\n✓ Entity-level cleaned data saved: Data/data_cleaned_entities.csv")

# ============================================================================
# STEP B: RELATIONSHIP-LEVEL CLEANING (Interaction Data)
# ============================================================================
print("\n" + "=" * 80)
print("STEP B: RELATIONSHIP-LEVEL CLEANING (Interaction Data)")
print("=" * 80)

# Rating columns (for predicting match)
rating_cols = ['like', 'prob', 'like_o', 'prob_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha']
rating_blocks = [['like', 'prob', 'like_o', 'prob_o'], ['pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha']]

# B1: Delete interactions with too many missing ratings (> 50% of block)
print("\n[B1] Removing interactions with > 50% missing in rating blocks...")
initial_rows = len(df_clean)
for block in rating_blocks:
    valid_cols = [c for c in block if c in df_clean.columns]
    missing_ratio = df_clean[valid_cols].isnull().sum(axis=1) / len(valid_cols)
    df_clean = df_clean[missing_ratio <= 0.5]
    print(f"   Block {valid_cols[:2]}...: removed {initial_rows - len(df_clean)} rows")
    initial_rows = len(df_clean)

# B2: Synchronize - Remove interactions of users that were dropped
print("\n[B2] Synchronizing interactions with cleaned entity list...")
valid_iids = set(df_clean['iid'].unique())
removed_due_sync = len(df_clean) - len(df_clean[df_clean['iid'].isin(valid_iids)])
df_clean = df_clean[df_clean['iid'].isin(valid_iids)].copy()
print(f"   Removed {removed_due_sync} orphan interactions")

# B3: Impute remaining missing values in interactions
print("\n[B3] Imputing remaining missing values in interactions...")
for col in df_clean.columns:
    if df_clean[col].dtype in ['int64', 'float64'] and df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        n_missing_before = df_clean[col].isnull().sum()
        df_clean[col] = df_clean[col].fillna(median_val)  # Use assignment
        # Only print if there were missing values
        # if n_missing_before > 0:
        #     print(f"   {col}: imputed {n_missing_before} with median")

# ============================================================================
# STEP C: FEATURE ENGINEERING (Order-Dependent)
# ============================================================================
print("\n" + "=" * 80)
print("STEP C: FEATURE ENGINEERING (Order-Dependent DAG)")
print("=" * 80)

# C1: Create age_gap BEFORE scaling (using raw age_map saved in A2)
print("\n[C1] Creating age_gap from raw age values...")
df_clean['age_gap'] = df_clean['iid'].map(age_map).fillna(df_clean['age'].median())
df_clean['age_gap'] = (df_clean['age_gap'] - df_clean['age_o']).abs()
print(f"   age_gap created: mean={df_clean['age_gap'].mean():.2f}, std={df_clean['age_gap'].std():.2f}")

# C2: Meaningful Aggregation - Combine 17 hobbies into 5 semantic groups
print("\n[C2] Aggregating 17 hobbies into 5 semantic feature groups...")
hobby_groups = {
    'fitness_sport': ['sports', 'tvsports', 'exercise'],
    'fine_arts': ['museums', 'art', 'reading', 'tv', 'theater'],
    'entertainment': ['movies', 'music', 'concerts', 'shopping'],
    'social_nightlife': ['dining', 'clubbing', 'gaming'],
    'outdoor_wellness': ['hiking', 'yoga']
}

for group_name, hobbies in hobby_groups.items():
    valid_hobbies = [h for h in hobbies if h in df_clean.columns]
    df_clean[group_name] = df_clean[valid_hobbies].mean(axis=1, skipna=True)
    print(f"   {group_name}: mean of {len(valid_hobbies)} hobbies")

# C3: Calculate hobby similarity (correlation between two people's hobby profiles)
print("\n[C3] Calculating hobby similarity (simplified version)...")
# For each interaction, check similarity in hobby interest patterns
hobby_cols_for_similarity = list(hobby_groups.keys())
df_clean['hobby_similarity'] = 0.0  # placeholder - would need person-level data to compute properly
print(f"   hobby_similarity placeholder added (would need person-specific data)")

# ============================================================================
# STEP D: SCALING (Differentiated Strategy)
# ============================================================================
print("\n" + "=" * 80)
print("STEP D: SCALING (Differentiated Strategy)")
print("=" * 80)

# D0: Final imputation BEFORE scaling (scaling cannot handle NaN)
print("\n[D0] Final imputation before scaling...")
n_imputed = 0
for col in df_clean.columns:
    if df_clean[col].dtype in ['int64', 'float64'] and df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        n_missing = df_clean[col].isnull().sum()
        df_clean[col] = df_clean[col].fillna(median_val)  # Use assignment, not inplace
        n_imputed += n_missing
if n_imputed > 0:
    print(f"   Imputed {n_imputed} total missing values across columns")
else:
    print(f"   No missing values found (all already imputed)")

# D1: MinMax Scaling [0,1] for 1-10 scale columns
print("\n[D1] MinMax Scaling [0,1] for bounded scales...")
minmax_cols = [c for c in ['like', 'prob', 'like_o', 'prob_o', 'age_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 
                            'pf_o_fun', 'pf_o_amb', 'pf_o_sha'] + hobby_cols_for_similarity
               if c in df_clean.columns]
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
for col in minmax_cols:
    df_clean[col] = scaler_minmax.fit_transform(df_clean[[col]])
print(f"   Applied MinMax to {len(minmax_cols)} columns")

# D2: StandardScaler (Z-score) for continuous columns
print("\n[D2] Standard Scaling (Z-score) for continuous variables...")
standard_cols = ['age', 'age_gap', 'age_o']
scaler_standard = StandardScaler()
for col in standard_cols:
    if col in df_clean.columns:
        df_clean[col] = scaler_standard.fit_transform(df_clean[[col]])
print(f"   Applied Standard to {len(standard_cols)} columns")

# ============================================================================
# STEP E: FINAL PREPARATION & VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP E: FINAL PREPARATION & VALIDATION")
print("=" * 80)

# E1: Select final columns for modeling
print("\n[E1] Selecting final feature set...")
final_features = ['iid', 'gender', 'age', 'match', 'like', 'prob', 'like_o', 'prob_o', 'age_o', 'age_gap', 
                  'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha',
                  'fitness_sport', 'fine_arts', 'entertainment', 'social_nightlife', 'outdoor_wellness',
                  'condtn', 'race', 'race_o']

final_features = [c for c in final_features if c in df_clean.columns]
df_final = df_clean[final_features].copy()
print(f"   Final feature count: {len(final_features)}")
print(f"   Final dataset shape before imputation: {df_final.shape}")

# E2: Final imputation for remaining missing values in final features
print("\n[E2] Final imputation for remaining missing values...")
missing_before = df_final.isnull().sum().sum()
for col in df_final.columns:
    if df_final[col].dtype in ['int64', 'float64'] and df_final[col].isnull().sum() > 0:
        median_val = df_final[col].median()
        n_missing = df_final[col].isnull().sum()
        df_final[col] = df_final[col].fillna(median_val)  # Use assignment
        if n_missing > 0:
            print(f"   {col}: imputed {n_missing} values")
missing_after = df_final.isnull().sum().sum()
print(f"   Total missing in final data: {missing_before} → {missing_after}")

# E3: Validation checks
print("\n[E3] Validation Checks...")
print(f"   ✓ Missing values: {df_final.isnull().sum().sum()}")
print(f"   ✓ Duplicate rows: {df_final.duplicated().sum()}")
print(f"   ✓ Target (match) distribution: {df_final['match'].value_counts().to_dict()}")
print(f"   ✓ Class balance: {(df_final['match'].sum() / len(df_final) * 100):.2f}% positive")

# E4: Save final dataset
df_final.to_csv('Data/data_final_v2.csv', index=False, encoding='utf-8')
print(f"\n✓ Final prepared data saved: Data/data_final_v2.csv")

# ============================================================================
# SUMMARY & STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n📊 Data Flow Summary:")
print(f"   Raw data:              {df.shape}")
print(f"   After entity cleaning: {df_clean.shape}")
print(f"   Final prepared data:   {df_final.shape}")

print(f"\n🔧 Principles Applied:")
print(f"   ✓ Entity vs Relationship Cleaning (impute users, delete interactions > 50%)")
print(f"   ✓ Order-Dependent FE (age_gap before scaling)")
print(f"   ✓ Scaling Strategy (MinMax [0,1] for bounded, Standard for continuous)")
print(f"   ✓ Meaningful Aggregation (17 hobbies -> 5 semantic groups)")
print(f"   ✓ Outlier Handling (IQR clip instead of drop)")
print(f"   ✓ Data Synchronization (removed {len(dropped_iids)} users + related interactions)")

print(f"\n📁 Output Files:")
print(f"   • Data/data_cleaned_entities.csv (entity-level cleaned)")
print(f"   • Data/data_final_v2.csv (final prepared for modeling)")

print("\n" + "=" * 80)
print("✓ TASK 03 COMPLETED")
print("=" * 80)
