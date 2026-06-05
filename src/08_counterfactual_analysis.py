import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        ('age_gap_calc', np.linspace(0, 20, 41), "Age Gap (Years)"),
        ('int_corr', np.linspace(-1, 1, 41), "Interest Correlation"),
        ('shar1_1', np.linspace(0, 100, 41), "Subject Pref for Shared Interests"),
        ('tvsports_o', np.linspace(0, 10, 21), "Partner's Interest in Watching Sports")
    ]
}

def load_assets():
    print(f"[1] Loading model and data...")
    winner = joblib.load(CONFIG['paths']['model_path'])
    df = pd.read_csv(CONFIG['paths']['input_data'])
    pipeline = winner['pipeline']
    threshold = winner['threshold']
    model_name = winner['name']
    X_all = df.drop(['iid', 'pid', 'match'], axis=1)
    return pipeline, threshold, df, X_all, model_name

def find_threshold_pair(pipeline, threshold, X_all):
    print(f"[2] Finding a pair close to threshold ({threshold:.4f})...")
    probs = pipeline.predict_proba(X_all)[:, 1]
    diffs = np.abs(probs - threshold)
    best_idx = np.argmin(diffs)
    print(f"   ✓ Selected row {best_idx} with prob {probs[best_idx]:.4f}")
    return X_all.iloc[[best_idx]]

def simulate_scenario(pipeline, typical_df, feature, values):
    probs = []
    temp_df = typical_df.copy()
    for val in values:
        temp_df[feature] = val
        prob = pipeline.predict_proba(temp_df)[0][1]
        probs.append(prob)
    return probs

def run_analysis():
    pipeline, threshold, df, X_all, model_name = load_assets()
    start_pair = find_threshold_pair(pipeline, threshold, X_all)
    
    print(f"[3] Simulating counterfactual scenarios for {model_name}...")
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    tipping_points = {}
    
    for i, (feat, vals, label) in enumerate(CONFIG['scenarios']):
        probs = simulate_scenario(pipeline, start_pair, feat, vals)
        
        axes[i].plot(vals, probs, marker='o', linestyle='-', color='teal', linewidth=2)
        axes[i].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
        
        # Find tipping point
        probs_arr = np.array(probs)
        crossings = np.where(np.diff(np.sign(probs_arr - threshold)))[0]
        
        if len(crossings) > 0:
            tp_val = vals[crossings[0]]
            tipping_points[feat] = tp_val
            axes[i].plot(tp_val, probs[crossings[0]], 'ro', markersize=10, fillstyle='none')
            axes[i].annotate(f'Tipping Point: {tp_val:.1f}', 
                             xy=(tp_val, probs[crossings[0]]), 
                             xytext=(tp_val+0.5, probs[crossings[0]]+0.05),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

        axes[i].set_title(f"Scenario: {label}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Match Probability")
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.suptitle(f"Counterfactual Analysis: How AI {model_name} Changes Its Mind (Sensitive Analysis)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(CONFIG['paths']['plot_path'])
    print(f"✓ Analysis complete. Plots saved to {CONFIG['paths']['plot_path']}")
    
    print("\n[4] Detected Tipping Points (Approximate):")
    for feat, val in tipping_points.items():
        print(f"    - {feat}: {val:.2f}")

if __name__ == "__main__":
    run_analysis()
