import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import importlib.util

# Dynamic import for module starting with a number
spec = importlib.util.spec_from_file_location("prep_module", "src/03_data_preparation.py")
prep_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep_module)

# Extract functions and config
load_data = prep_module.load_data
normalize_survey_scales = prep_module.normalize_survey_scales
create_user_profiles = prep_module.create_user_profiles
build_dyadic_dataset = prep_module.build_dyadic_dataset
PREP_CONFIG = prep_module.CONFIG

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'paths': {
        'input_data': 'Data/data_final_v2.csv',
        'model_path': 'models/winner_model.joblib',
        'plot_path': 'plots/counterfactual_scenarios.png'
    },
    'scenarios': [
        ('age_gap_calc', np.linspace(-1.5, 3.5, 51), "Age Gap (Years) - Unscaled", 'standard'),
        ('int_corr', np.linspace(-1, 1, 41), "Interest Correlation", 'none'),
        ('art_gap', np.linspace(0, 1, 21), "Art Interest Gap (1-10 scale)", 'minmax'),
        ('attr_surplus_51_s', np.linspace(-5, 5, 21), "Attractiveness Surplus (Partner exceeds me)", 'standard')
    ]
}

def load_scaling_params():
    print("[0] Extracting scaling parameters from raw dyadic data...")
    df_raw = load_data(PREP_CONFIG['paths']['raw_data'])
    df_norm = normalize_survey_scales(df_raw)
    user_profiles, dropped_iids = create_user_profiles(df_norm, PREP_CONFIG)
    df_pair = build_dyadic_dataset(df_raw, user_profiles, dropped_iids, PREP_CONFIG)
    
    params = {}
    
    # Age Gap (Standard Scaled)
    age_gap_raw = (df_pair['age'] - df_pair['age_o']).abs()
    params['age_gap_calc'] = {'mean': age_gap_raw.mean(), 'std': age_gap_raw.std()}
    
    # Hobby Gap (MinMax Scaled originally from 1-10)
    params['minmax_gap'] = {'min': 0, 'max': 9}
    
    # Surplus (Standard Scaled)
    surplus_raw = df_pair['attr5_1_o'] - df_pair['attr1_1']
    params['surplus'] = {'mean': surplus_raw.mean(), 'std': surplus_raw.std()}
    
    return params

def inverse_scale(val, scale_type, params, feat_name):
    if scale_type == 'standard' and 'age' in feat_name:
        return (val * params['age_gap_calc']['std']) + params['age_gap_calc']['mean']
    elif scale_type == 'standard' and 'surplus' in feat_name:
        return (val * params['surplus']['std']) + params['surplus']['mean']
    elif scale_type == 'minmax':
        return val * params['minmax_gap']['max']
    return val

def load_assets():
    print(f"[1] Loading model and data...")
    winner = joblib.load(CONFIG['paths']['model_path'])
    df = pd.read_csv(CONFIG['paths']['input_data'])
    pipeline = winner['pipeline']
    threshold = winner['threshold']
    model_name = winner['name']
    
    train_features = list(pipeline.feature_names_in_)
    print(f"   ✓ Model expects {len(train_features)} features.")
    
    X_all = df[train_features].astype(float)
    return pipeline, threshold, df, X_all, train_features, model_name

def find_threshold_pair(pipeline, threshold, X_all):
    print(f"[2] Finding a pair close to threshold ({threshold:.4f})...")
    probs = pipeline.predict_proba(X_all)[:, 1]
    diffs = np.abs(probs - threshold)
    best_idx = np.argmin(diffs)
    print(f"   ✓ Selected row {best_idx} with prob {probs[best_idx]:.4f}")
    return X_all.iloc[[best_idx]].copy()

def simulate_scenario(pipeline, start_row, feature, values, train_features):
    probs = []
    temp_df = start_row.copy()
    for val in values:
        temp_df[feature] = val
        input_data = temp_df[train_features]
        prob = pipeline.predict_proba(input_data)[0][1]
        probs.append(prob)
    return probs

def run_analysis():
    scale_params = load_scaling_params()
    pipeline, threshold, df, X_all, train_features, model_name = load_assets()
    start_pair = find_threshold_pair(pipeline, threshold, X_all)
    
    print(f"[3] Simulating counterfactual scenarios for {model_name}...")
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    tipping_points = {}
    
    for i, (feat, vals, label, scale_type) in enumerate(CONFIG['scenarios']):
        # X_vals_plot will be real world numbers, vals are scaled numbers for the model
        x_vals_plot = [inverse_scale(v, scale_type, scale_params, feat) for v in vals]
        probs = simulate_scenario(pipeline, start_pair, feat, vals, train_features)
        
        axes[i].plot(x_vals_plot, probs, marker='o', linestyle='-', color='teal', linewidth=2)
        axes[i].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
        
        probs_arr = np.array(probs)
        crossings = np.where(np.diff(np.sign(probs_arr - threshold)))[0]
        
        if len(crossings) > 0:
            tp_idx = crossings[0]
            tp_val_real = x_vals_plot[tp_idx]
            tipping_points[feat] = tp_val_real
            axes[i].plot(tp_val_real, probs[tp_idx], 'ro', markersize=10, fillstyle='none')
            axes[i].annotate(f'Tipping Point: {tp_val_real:.1f}', 
                             xy=(tp_val_real, probs[tp_idx]), 
                             xytext=(tp_val_real + (max(x_vals_plot)-min(x_vals_plot))*0.05, probs[tp_idx]+0.05),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

        axes[i].set_title(f"Scenario: {label}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Match Probability")
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.suptitle(f"Counterfactual Analysis: How AI {model_name} Changes Its Mind (UNSCALED VIEW)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(CONFIG['paths']['plot_path'])
    print(f"✓ Analysis complete. Plots saved to {CONFIG['paths']['plot_path']}")
    
    print("\n[4] Detected Tipping Points (Real-world scale):")
    for feat, val in tipping_points.items():
        print(f"    - {feat}: {val:.2f}")

if __name__ == "__main__":
    run_analysis()
