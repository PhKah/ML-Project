import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# [KNOWLEDGE-DRIVEN] CONFIGURATION - INTEGRATING COMPATIBILITY GAPS
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
    },
    'scaling': {
        'minmax': [
            'int_corr', 'exphappy', 'expnum',
            'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
            'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1',
            'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
            'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1',
            'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1',
            'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
            'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
            'movies', 'concerts', 'music', 'shopping', 'yoga'
        ],
        'standard': ['age']
    },
    'tipping_points': {
        'age_gap': 0.5,       # Refined from src/08 Counterfactual Analysis
        'interest_corr': 0.30 # Refined from src/08 Counterfactual Analysis
    }
}

print("=" * 80)
print("TASK 03: [REFAC] COMPATIBILITY-DRIVEN DATA PREPARATION")
print("=" * 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data(path):
    print(f"\n[1] Loading data from {path}...")
    return pd.read_csv(path, encoding='ISO-8859-1')

def normalize_survey_scales(df):
    """Normalize Waves 6-9 from 1-10 to 100pt scale for comparison."""
    survey_groups = {
        'pref': ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'],
        'partner': ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1'],
        'fellow': ['attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1']
    }
    mask_69 = df['wave'].between(6, 9)
    if mask_69.any():
        print(f"   Normalizing {mask_69.sum()} rows for waves 6-9 (1-10 -> 100pt)")
        for cols in survey_groups.values():
            valid_cols = [c for c in cols if c in df.columns]
            row_sum = df.loc[mask_69, valid_cols].sum(axis=1).replace(0, 1)
            for col in valid_cols:
                df.loc[mask_69, col] = (df.loc[mask_69, col] / row_sum) * 100
    return df

def create_user_profiles(df, config):
    """Extract and clean entity-level profiles (1 row per iid)."""
    print("\n" + "=" * 40)
    print("STEP A: CREATE CLEAN USER PROFILES")
    print("=" * 40)
    
    cols = config['features']['entity_cols']
    profiles = df.groupby('iid')[cols].first()
    
    # A1: Handle Missingness
    missing_count = profiles.isnull().sum(axis=1)
    dropped_iids = missing_count[missing_count >= 20].index.tolist()
    profiles = profiles.drop(dropped_iids)
    print(f"   Dropped {len(dropped_iids)} users with >= 20 missing profile values")
    
    # A2: Impute & Clip
    for col in profiles.columns:
        if profiles[col].dtype in ['int64', 'float64']:
            profiles[col] = profiles[col].fillna(profiles[col].median())
            if col not in ['gender', 'race', 'goal', 'career_c']:
                Q1, Q3 = profiles[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                profiles[col] = profiles[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
                
    return profiles, dropped_iids

def build_dyadic_dataset(df_raw, user_profiles, dropped_iids, config):
    """Join subject and partner profiles to create Pair Profiles."""
    print("\n" + "=" * 40)
    print("STEP B: BUILD PAIR PROFILES (JOIN)")
    print("=" * 40)
    
    interactions = df_raw[config['features']['interaction_cols']].copy()
    interactions = interactions[~interactions['iid'].isin(dropped_iids)]
    interactions = interactions[~interactions['pid'].isin(dropped_iids)]
    
    initial_len = len(interactions)
    interactions = interactions[interactions['pid'].isin(user_profiles.index)]
    print(f"   Referential Integrity: Dropped {initial_len - len(interactions)} orphan interactions")
    
    df_pair = interactions.merge(user_profiles, left_on='iid', right_index=True, how='left')
    df_pair = df_pair.merge(user_profiles, left_on='pid', right_index=True, how='left', suffixes=('', '_o'))
    
    print(f"   Merged Subject & Partner profiles. New shape: {df_pair.shape}")
    return df_pair

def calculate_dyadic_features(df, config):
    """Compute features including compatibility gaps (The Principle of Similarity)."""
    print("\n" + "=" * 40)
    print("STEP C: COMPUTE DYADIC FEATURES (COMPATIBILITY GAPS)")
    print("=" * 40)
    
    # C1: Raw Age Gap
    df['age_gap_calc'] = (df['age'] - df['age_o']).abs()
    
    # C2: Hobby Gaps (Similarity in Lifestyle)
    hobbies = [
        'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
        'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
        'movies', 'concerts', 'music', 'shopping', 'yoga'
    ]
    for h in hobbies:
        if h in df.columns and f"{h}_o" in df.columns:
            df[f'{h}_gap'] = (df[h] - df[f"{h}_o"]).abs()
            
    # C3: Expectation Gaps (Similarity in Values)
    gap_map = {
        'attr': ('attr1_1', 'attr3_1_o'),
        'sinc': ('sinc1_1', 'sinc3_1_o'),
        'intel': ('intel1_1', 'intel3_1_o'),
        'fun': ('fun1_1', 'fun3_1_o'),
        'amb': ('amb1_1', 'amb3_1_o'),
        'shar': ('shar1_1', 'shar3_1_o')
    }
    for key, (pref, self_r) in gap_map.items():
        if pref in df.columns and self_r in df.columns:
            df[f'{key}_exp_gap'] = (df[pref] - df[self_r]).abs()
    
    # C4: Tipping Point Indicators from GĐ 6
    tp = config['tipping_points']
    df['is_age_match'] = (df['age_gap_calc'] <= tp['age_gap']).astype(int)
    df['is_interest_match'] = (df['int_corr'] >= tp['interest_corr']).astype(int)
    df['match_synergy'] = df['is_age_match'] * df['is_interest_match']
    
    print(f"   ✓ Integrated 23 Gap features + 3 Tipping point indicators.")
    return df

def apply_scaling(df, config):
    print("\n" + "=" * 40)
    print("STEP D: SYSTEMATIC SCALING")
    print("=" * 40)
    
    minmax_list = []
    standard_list = []
    
    # Identify which columns need which scaling
    for col in config['scaling']['minmax']:
        if col in df.columns: minmax_list.append(col)
        if f"{col}_o" in df.columns: minmax_list.append(f"{col}_o")
    
    # Gap features are derived from MinMax [0,1], so they stay in MinMax
    gap_cols = [c for c in df.columns if '_gap' in c or '_exp_gap' in c]
    minmax_list.extend(gap_cols)
    
    for col in config['scaling']['standard']:
        if col in df.columns: standard_list.append(col)
        if f"{col}_o" in df.columns: standard_list.append(f"{col}_o")
        
    if 'age_gap_calc' in df.columns: standard_list.append('age_gap_calc')
    
    # Ensure lists are unique and exist in df
    minmax_list = list(set([c for c in minmax_list if c in df.columns]))
    standard_list = list(set([c for c in standard_list if c in df.columns]))
    
    # Indicator features (0/1) and Identifiers should NOT be scaled
    passthrough_list = [c for c in df.columns if c not in minmax_list + standard_list]
    
    ct = ColumnTransformer([
        ('minmax', MinMaxScaler(), minmax_list),
        ('standard', StandardScaler(), standard_list)
    ], remainder='passthrough')
    
    vals = ct.fit_transform(df)
    cols_after = minmax_list + standard_list + passthrough_list
    
    return pd.DataFrame(vals, columns=cols_after, index=df.index)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    df_raw = load_data(CONFIG['paths']['raw_data'])
    df_norm = normalize_survey_scales(df_raw)
    
    # 1. Create User Profiles
    user_profiles, dropped_iids = create_user_profiles(df_norm, CONFIG)
    
    # 2. Join into Pair Profiles
    df_pair = build_dyadic_dataset(df_raw, user_profiles, dropped_iids, CONFIG)
    
    # 3. Compute Compatibility (Including Knowledge-based features)
    df_fe = calculate_dyadic_features(df_pair, CONFIG)
    
    # 4. Final Impute & Scaling
    df_fe = df_fe.fillna(df_fe.median())
    df_scaled = apply_scaling(df_fe, CONFIG)
    
    # 5. Final Cleanup
    if 'wave' in df_scaled.columns: df_scaled = df_scaled.drop(columns='wave')
    
    df_scaled.to_csv(CONFIG['paths']['final_data'], index=False)
    
    print(f"\n✓ Process completed. Final data: {df_scaled.shape}")
    print(f"✓ Compatibility Gaps & Tipping points integrated into the pipeline.")
    print(f"✓ Saved to: {CONFIG['paths']['final_data']}")
