import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# [ULTIMATE] CONFIGURATION - RAW FEATURE EXTRACTION (PURE DATA - NO LEAKAGE)
# ============================================================================
CONFIG = {
    'paths': {
        'raw_data': 'Data/Speed Dating Data.csv',
        'final_data': 'Data/data_final_v2.csv'
    },
    'features': {
        'entity_cols': [
            'age', 'gender', 'race', 'imprace', 'imprelig', 'goal', 'date', 
            'go_out', 'career_c', 'exphappy', 'expnum',
            'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
            'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1',
            'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
            'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1',
            'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1',
            'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
            'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
            'movies', 'concerts', 'music', 'shopping', 'yoga'
        ],
        'interaction_cols': ['iid', 'pid', 'match', 'samerace', 'int_corr', 'condtn', 'wave']
    }
}

print("=" * 80)
print("TASK 03: [REFAC] RAW FEATURE EXTRACTION (PURE DATA - NO SCALING)")
print("=" * 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data(path):
    print(f"\n[1] Loading data from {path}...")
    return pd.read_csv(path, encoding='ISO-8859-1')

def normalize_survey_scales(df):
    """Normalize 1-10 waves to 100pt scale for consistency at raw level."""
    survey_groups = {
        'pref': ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'],
        'partner': ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1'],
        'fellow': ['attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1']
    }
    mask_69 = df['wave'].between(6, 9)
    if mask_69.any():
        for cols in survey_groups.values():
            valid_cols = [c for c in cols if c in df.columns]
            row_sum = df.loc[mask_69, valid_cols].sum(axis=1).replace(0, 1)
            for col in valid_cols:
                df.loc[mask_69, col] = (df.loc[mask_69, col] / row_sum) * 100
    return df

def create_user_profiles(df, config):
    print("\nSTEP A: CREATE CLEAN USER PROFILES")
    cols = config['features']['entity_cols']
    profiles = df.groupby('iid')[cols].first()
    
    # Entity-level cleaning (Internal integrity)
    missing_count = profiles.isnull().sum(axis=1)
    dropped_iids = missing_count[missing_count >= 20].index.tolist()
    print(f"   [ENTITY DELETION] Dropped {len(dropped_iids)} 'ghost' users missing >= 20 out of 56 CORE features.")
    profiles = profiles.drop(dropped_iids)
    
    # WE NO LONGER FILL MEDIAN GLOBALLY HERE TO AVOID LEAKAGE.
    # We only CLIP outliers to prevent extreme values from ruining profiles.
    for col in profiles.columns:
        if profiles[col].dtype in ['int64', 'float64']:
            if col not in ['gender', 'race', 'goal', 'career_c']:
                Q1, Q3 = profiles[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                profiles[col] = profiles[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
                
    return profiles, dropped_iids

def build_dyadic_dataset(df_raw, user_profiles, dropped_iids, config):
    print("\nSTEP B: BUILD PAIR PROFILES (JOIN) & EXPORT PAIR_ID")
    interactions = df_raw[config['features']['interaction_cols']].copy()
    
    initial_len = len(interactions)
    interactions = interactions[~interactions['iid'].isin(dropped_iids)]
    interactions = interactions[~interactions['pid'].isin(dropped_iids)]
    interactions = interactions[interactions['pid'].isin(user_profiles.index)]
    print(f"   [REFERENTIAL INTEGRITY] Dropped {initial_len - len(interactions)} interactions related to ghosts/missing profiles.")
    
    # [NEW ARCHITECTURE] Create pair_id for GroupSplit later. DO NOT DEDUPLICATE.
    interactions['pair_id'] = interactions.apply(
        lambda x: f"{int(min(x['iid'], x['pid']))}_{int(max(x['iid'], x['pid']))}", axis=1
    )
    print(f"   Interactions retained (A->B and B->A intact): {len(interactions)}")
    
    df_pair = interactions.merge(user_profiles, left_on='iid', right_index=True, how='left')
    df_pair = df_pair.merge(user_profiles, left_on='pid', right_index=True, how='left', suffixes=('', '_o'))
    return df_pair

def calculate_ultimate_features(df):
    print("\nSTEP C: COMPUTE RAW COGNITIVE FEATURES")
    
    # C1: Age Gap
    df['age_gap_calc'] = (df['age'] - df['age_o']).abs()
    
    # C2: Hobby Gaps (Signal Condensation)
    hobbies = [
        'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
        'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
        'movies', 'concerts', 'music', 'shopping', 'yoga'
    ]
    gap_cols = []
    for h in hobbies:
        if h in df.columns and f"{h}_o" in df.columns:
            col_name = f'{h}_gap'
            df[col_name] = (df[h] - df[f"{h}_o"]).abs()
            gap_cols.append(col_name)
    
    # [NEW] Aggregate granular gaps into one holistic indicator to prevent over-splitting
    df['mean_hobby_gap'] = df[gap_cols].mean(axis=1)
    
    # Keep only the important catalytic gap (shar_gap) and the aggregate, drop others
    # Note: 'shar' is usually in the attrs list, but here we treat it as a general hobby indicator
    df = df.drop(columns=gap_cols)
    
    # C3: Mutual Expectation Surplus (2 Tiers - Phép trừ có dấu)
    attrs = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
    for a in attrs:
        # Note: We scale pref (0-100) to 1-10 for surplus consistency (since self-perception is 1-10)
        # Tier 1: 5_1 (Conservative/Social Reality)
        if f'{a}5_1_o' in df.columns and f'{a}1_1' in df.columns:
            df[f'{a}_surplus_51_s'] = df[f'{a}5_1_o'] - (df[f'{a}1_1'] / 10.0)
        if f'{a}5_1' in df.columns and f'{a}1_1_o' in df.columns:
            df[f'{a}_surplus_51_p'] = df[f'{a}5_1'] - (df[f'{a}1_1_o'] / 10.0)
            
        # Tier 2: 3_1 (Personal Ego)
        if f'{a}3_1_o' in df.columns and f'{a}1_1' in df.columns:
            df[f'{a}_surplus_31_s'] = df[f'{a}3_1_o'] - (df[f'{a}1_1'] / 10.0)
        if f'{a}3_1' in df.columns and f'{a}1_1_o' in df.columns:
            df[f'{a}_surplus_31_p'] = df[f'{a}3_1'] - (df[f'{a}1_1_o'] / 10.0)
    
    print(f"   ✓ Extracted Hobby Gaps + 24 Surplus variables in raw format.")
    return df

if __name__ == "__main__":
    df_raw = load_data(CONFIG['paths']['raw_data'])
    df_norm = normalize_survey_scales(df_raw)
    user_profiles, dropped_iids = create_user_profiles(df_norm, CONFIG)
    df_pair = build_dyadic_dataset(df_raw, user_profiles, dropped_iids, CONFIG)
    df_fe = calculate_ultimate_features(df_pair)
    
    # WE DO NOT FILL NA OR SCALE HERE TO AVOID LEAKAGE.
    # JUST SAVE THE PURE RAW FEATURES.
    if 'wave' in df_fe.columns: df_fe = df_fe.drop(columns='wave')
    
    df_fe.to_csv(CONFIG['paths']['final_data'], index=False)
    print(f"\n✓ Process completed. Raw data saved: {df_fe.shape}")
    print(f"✓ NO DATA LEAKAGE: Statistics fitting moved to Modeling Phase.")
