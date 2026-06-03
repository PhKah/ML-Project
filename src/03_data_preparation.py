import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# [UPDATED] CONFIGURATION SECTION - DATA-DRIVEN & PRE-MATCH
# ============================================================================
CONFIG = {
    'paths': {
        'raw_data': 'Data/Speed Dating Data.csv',
        'intermediate_data': 'Data/data_cleaned_entities.csv',
        'final_data': 'Data/data_final_v2.csv'
    },
    'features': {
        'entity_cols': [
            'age', 'gender', 'race', 'imprace', 'imprelig', 'goal', 'date', 
            'go_out', 'career_c', 'exphappy', 'expnum', 'wave',
            'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
            'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1',
            'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
            'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1',
            'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1',
            'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
            'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
            'movies', 'concerts', 'music', 'shopping', 'yoga'
        ],
        'relationship_cols': ['samerace', 'int_corr', 'match', 'age_o', 'race_o'],
        'hobby_groups': {
            'fitness_sport': ['sports', 'tvsports', 'exercise'],
            'fine_arts': ['museums', 'art', 'reading', 'tv', 'theater'],
            'entertainment': ['movies', 'music', 'concerts', 'shopping'],
            'social_nightlife': ['dining', 'clubbing', 'gaming'],
            'outdoor_wellness': ['hiking', 'yoga']
        },
        'final_selection': [
            'iid', 'pid', 'gender', 'age', 'match', 'samerace', 'int_corr',
            'age_o', 'age_gap', 'race', 'race_o', 'goal', 'date', 'go_out', 
            'exphappy', 'expnum',
            'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
            'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1',
            'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
            'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1',
            'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1',
            'fitness_sport', 'fine_arts', 'entertainment', 'social_nightlife', 'outdoor_wellness',
            'condtn'
        ]
    },
    'scaling': {
        'minmax': [
            'int_corr', 'fitness_sport', 'fine_arts', 'entertainment', 
            'social_nightlife', 'outdoor_wellness', 'exphappy', 'expnum',
            'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
            'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1',
            'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
            'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1',
            'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1'
        ],
        'standard': ['age', 'age_gap', 'age_o']
    }
}

print("=" * 80)
print("TASK 03: SYSTEMATIC DATA PREPARATION (PHASE 2)")
print("=" * 80)

# ============================================================================
# FUNCTIONS FOR MODULARITY
# ============================================================================

def load_data(path):
    print(f"\n[1] Loading data from {path}...")
    return pd.read_csv(path, encoding='ISO-8859-1')

def normalize_survey_scales(df):
    """
    Waves 6-9 use 1-10 scale. Others use 100-point allocation.
    We convert 1-10 to 'percentage' (sum to 100) to ensure consistency.
    """
    print("\n" + "=" * 40)
    print("STEP B0: NORMALIZE MIXED SCALES (WAVES 6-9)")
    print("=" * 40)
    
    survey_groups = {
        'pref': ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'],
        'partner': ['attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1'],
        'fellow': ['attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1']
    }
    
    mask_69 = df['wave'].between(6, 9)
    if mask_69.any():
        print(f"   Normalizing {mask_69.sum()} rows for waves 6-9")
        for name, cols in survey_groups.items():
            valid_cols = [c for c in cols if c in df.columns]
            # Sum for each person
            row_sum = df.loc[mask_69, valid_cols].sum(axis=1)
            # Avoid division by zero
            row_sum = row_sum.replace(0, 1)
            # Normalize to 100
            for col in valid_cols:
                df.loc[mask_69, col] = (df.loc[mask_69, col] / row_sum) * 100
                
    return df

def entity_cleaning(df, config):
    print("\n" + "=" * 40)
    print("STEP A: ENTITY-LEVEL CLEANING")
    print("=" * 40)
    cols = config['features']['entity_cols']
    
    # A1: Remove users with excess missing
    # Group by iid to evaluate the static entity profile
    iid_profile = df.groupby('iid')[cols].first()
    entity_missing = iid_profile.isnull().sum(axis=1)
    dropped_iids = entity_missing[entity_missing >= 15].index.tolist() # Relaxed threshold
    
    df_clean = df[~df['iid'].isin(dropped_iids)].copy()
    print(f"   Dropped {len(dropped_iids)} users with >= 15 missing values in entity profile")

    # A1.1: Referential Integrity (Drop interactions where partner was dropped)
    initial_rows = len(df_clean)
    df_clean = df_clean[~df_clean['pid'].isin(dropped_iids)]
    print(f"   Dropped {initial_rows - len(df_clean)} interactions due to referential integrity (dropped partners)")

    # A2: Save age map for age_gap
    age_map = df_clean[['iid', 'age']].drop_duplicates().set_index('iid')['age'].to_dict()

    # A3 & A4: Impute & Clip
    for col in cols:
        if col in df_clean.columns:
            # Impute
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                # Clip (Exclude identifiers and categorical codes)
                if col not in ['gender', 'race', 'goal', 'career_c', 'iid', 'pid']:
                    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    df_clean[col] = df_clean[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    
    return df_clean, age_map

def relationship_cleaning(df, config):
    print("\n" + "=" * 40)
    print("STEP B: RELATIONSHIP-LEVEL CLEANING")
    print("=" * 40)
    
    # B1: Filter by relationship data missingness
    rel_cols = [c for c in config['features']['relationship_cols'] if c in df.columns]
    missing_ratio = df[rel_cols].isnull().sum(axis=1) / len(rel_cols)
    df = df[missing_ratio <= 0.5]
    print(f"   Dropped rows with > 50% relationship data missing")
    
    # B3: Impute remaining
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def feature_engineering(df, age_map, config):
    print("\n" + "=" * 40)
    print("STEP C: FEATURE ENGINEERING")
    print("=" * 40)
    
    # C1: age_gap
    # Ensure age_gap is absolute difference
    df['age_gap'] = df['iid'].map(age_map)
    # If age not in map, use current row age as fallback before calculation
    df['age_gap'] = df['age_gap'].fillna(df['age'])
    df['age_gap'] = (df['age_gap'] - df['age_o']).abs()

    # C2: Aggregation
    for group, hobbies in config['features']['hobby_groups'].items():
        valid_hobbies = [h for h in hobbies if h in df.columns]
        df[group] = df[valid_hobbies].mean(axis=1)
        
    return df

def apply_scaling(df, config):
    print("\n" + "=" * 40)
    print("STEP D: SYSTEMATIC SCALING")
    print("=" * 40)
    
    # Check if columns exist in df
    minmax_cols = [c for c in config['scaling']['minmax'] if c in df.columns]
    standard_cols = [c for c in config['scaling']['standard'] if c in df.columns]
    
    ct = ColumnTransformer([
        ('minmax', MinMaxScaler(), minmax_cols),
        ('standard', StandardScaler(), standard_cols)
    ], remainder='passthrough')
    
    # Process
    df_scaled_values = ct.fit_transform(df)
    
    # Reconstruct DataFrame with correct columns
    cols_after_ct = minmax_cols + standard_cols
    all_cols = df.columns.tolist()
    remainder_cols = [c for c in all_cols if c not in cols_after_ct]
    
    df_scaled = pd.DataFrame(df_scaled_values, columns=cols_after_ct + remainder_cols, index=df.index)
    
    return df_scaled

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    df_raw = load_data(CONFIG['paths']['raw_data'])
    
    # NEW: Normalize scales for waves 6-9 before cleaning
    df_norm = normalize_survey_scales(df_raw)
    
    df_entity, age_map = entity_cleaning(df_norm, CONFIG)
    df_rel = relationship_cleaning(df_entity, CONFIG)
    df_fe = feature_engineering(df_rel, age_map, CONFIG)
    
    # Final Selection
    final_cols = [c for c in CONFIG['features']['final_selection'] if c in df_fe.columns]
    df_final_prep = df_fe[final_cols].copy()
    
    # Final Impute Check
    df_final_prep = df_final_prep.fillna(df_final_prep.median())
    
    df_scaled = apply_scaling(df_final_prep, CONFIG)
    
    # Save final results
    df_scaled.to_csv(CONFIG['paths']['final_data'], index=False)
    
    print(f"\n✓ Process completed. Final data: {df_scaled.shape}")
    print(f"✓ Mixed scales normalized (Waves 6-9 -> 100pt)")
    print(f"✓ Pre-match strategy applied (Anti-leakage)")
    print(f"✓ Saved to: {CONFIG['paths']['final_data']}")
