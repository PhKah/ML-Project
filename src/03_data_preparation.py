import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# [ULTIMATE] CONFIGURATION - MULTI-LEVEL COGNITIVE INTEGRATION
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
    }
}

print("=" * 80)
print("TASK 03: [REFAC] ULTIMATE COGNITIVE DATA PREPARATION")
print("=" * 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data(path):
    print(f"\n[1] Loading data from {path}...")
    return pd.read_csv(path, encoding='ISO-8859-1')

def normalize_survey_scales(df):
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
    
    missing_count = profiles.isnull().sum(axis=1)
    dropped_iids = missing_count[missing_count >= 20].index.tolist()
    print(f"   [ENTITY DELETION] Dropped {len(dropped_iids)} 'ghost' users with >= 20 missing features.")
    profiles = profiles.drop(dropped_iids)
    
    for col in profiles.columns:
        if profiles[col].dtype in ['int64', 'float64']:
            profiles[col] = profiles[col].fillna(profiles[col].median())
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

def calculate_ultimate_features(df, config):
    print("\nSTEP C: COMPUTE ULTIMATE COGNITIVE FEATURES")
    
    # C1: Age Gap
    df['age_gap_calc'] = (df['age'] - df['age_o']).abs()
    
    # C2: Hobby Gaps (Similarity - absolute)
    hobbies = [
        'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
        'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
        'movies', 'concerts', 'music', 'shopping', 'yoga'
    ]
    for h in hobbies:
        if h in df.columns and f"{h}_o" in df.columns:
            df[f'{h}_gap'] = (df[h] - df[f"{h}_o"]).abs()
            
    # C3: Mutual Expectation Surplus (2 Tiers)
    attrs = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
    
    for a in attrs:
        # Tier 1: 5_1 (Conservative/Social Reality)
        # Surplus S: How much Partner exceeds my expectations
        if f'{a}5_1_o' in df.columns and f'{a}1_1' in df.columns:
            df[f'{a}_surplus_51_s'] = df[f'{a}5_1_o'] - df[f'{a}1_1']
        # Surplus P: How much I exceed Partner's expectations
        if f'{a}5_1' in df.columns and f'{a}1_1_o' in df.columns:
            df[f'{a}_surplus_51_p'] = df[f'{a}5_1'] - df[f'{a}1_1_o']
            
        # Tier 2: 3_1 (Personal Ego)
        # Surplus S: How much Partner thinks they exceed my expectations
        if f'{a}3_1_o' in df.columns and f'{a}1_1' in df.columns:
            df[f'{a}_surplus_31_s'] = df[f'{a}3_1_o'] - df[f'{a}1_1']
        # Surplus P: How much I think I exceed Partner's expectations
        if f'{a}3_1' in df.columns and f'{a}1_1_o' in df.columns:
            df[f'{a}_surplus_31_p'] = df[f'{a}3_1'] - df[f'{a}1_1_o']
    
    print(f"   ✓ Integrated Hobby Gaps + 24 Surplus variables. (Indicators removed)")
    return df

def apply_scaling(df, config):
    print("\nSTEP D: SYSTEMATIC SCALING")
    minmax_list = list(set([c for c in config['scaling']['minmax'] if c in df.columns]))
    standard_list = list(set([c for c in config['scaling']['standard'] if c in df.columns]))
    
    # Add '_o' columns
    o_minmax = [f"{c}_o" for c in minmax_list if f"{c}_o" in df.columns]
    o_standard = [f"{c}_o" for c in standard_list if f"{c}_o" in df.columns]
    minmax_list.extend(o_minmax)
    standard_list.extend(o_standard)
    
    # Add new engineered continuous features
    eng_cont = [c for c in df.columns if '_gap' in c or '_surplus_' in c or 'age_gap_calc' in c]
    standard_list.extend(eng_cont)
    
    minmax_list = list(set(minmax_list))
    standard_list = list(set(standard_list))
    
    # Make sure we don't try to scale string columns like pair_id
    passthrough_list = [c for c in df.columns if c not in minmax_list + standard_list]
    
    ct = ColumnTransformer([
        ('minmax', MinMaxScaler(), minmax_list),
        ('standard', StandardScaler(), standard_list)
    ], remainder='passthrough')
    
    vals = ct.fit_transform(df)
    return pd.DataFrame(vals, columns=minmax_list + standard_list + passthrough_list, index=df.index)

if __name__ == "__main__":
    df_raw = load_data(CONFIG['paths']['raw_data'])
    df_norm = normalize_survey_scales(df_raw)
    user_profiles, dropped_iids = create_user_profiles(df_norm, CONFIG)
    df_pair = build_dyadic_dataset(df_raw, user_profiles, dropped_iids, CONFIG)
    df_fe = calculate_ultimate_features(df_pair, CONFIG)
    df_fe = df_fe.fillna(df_fe.median(numeric_only=True))
    df_scaled = apply_scaling(df_fe, CONFIG)
    if 'wave' in df_scaled.columns: df_scaled = df_scaled.drop(columns='wave')
    df_scaled.to_csv(CONFIG['paths']['final_data'], index=False)
    print(f"\n✓ Process completed. Final data: {df_scaled.shape}")
