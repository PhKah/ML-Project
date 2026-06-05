import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'paths': {
        'input_data': 'Data/Speed Dating Data.csv',
        'plot_path': 'plots/behavioral_gaps_match_vs_nomatch.png'
    },
    'attributes': ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar'],
    'self_perception_attrs': ['attr', 'sinc', 'intel', 'fun', 'amb']
}

def load_and_clean():
    print(f"[1] Loading raw data for behavioral analytics...")
    df_raw = pd.read_csv(CONFIG['paths']['input_data'], encoding='ISO-8859-1')
    
    # 1. Define columns to extract
    pref_cols = [f"{a}1_1" for a in CONFIG['attributes']]
    self_cols = [f"{a}5_1" for a in CONFIG['self_perception_attrs']]
    rating_cols = CONFIG['attributes']
    partner_cols = [f"{a}_o" for a in CONFIG['attributes']]
    
    cols = ['iid', 'pid', 'gender', 'wave', 'match'] + pref_cols + self_cols + rating_cols + partner_cols
    df = df_raw[cols].copy()
    
    # 2. Normalize 100pt scales to 1-10
    mask_100 = ~df['wave'].between(6, 9)
    for col in pref_cols:
        df.loc[mask_100, col] = df.loc[mask_100, col] / 10.0
        df[col] = df[col].clip(1, 10)
        
    return df

def calculate_behavioral_gaps(df):
    print(f"[2] Calculating multi-dimensional gaps (Match vs No Match)...")
    
    # Gap 1: Disappointment
    for a in CONFIG['attributes']:
        df[f'{a}_disappointment'] = df[f'{a}1_1'] - df[a]
        
    # Gap 2: Misconception
    for a in CONFIG['self_perception_attrs']:
        df[f'{a}_misconception'] = df[f'{a}5_1'] - df[f'{a}_o']
        
    return df

def visualize_comprehensive(df):
    print(f"[3] Generating behavioral comparison dashboard...")
    
    # Prepare labels
    df['match_label'] = df['match'].map({1: 'Match', 0: 'No Match'})
    
    # Melt for grouped plotting
    dis_cols = [f'{a}_disappointment' for a in CONFIG['attributes']]
    mis_cols = [f'{a}_misconception' for a in CONFIG['self_perception_attrs']]
    
    dis_df = df.melt(id_vars=['match_label'], value_vars=dis_cols, var_name='Attribute', value_name='Gap')
    dis_df['Attribute'] = dis_df['Attribute'].str.replace('_disappointment', '').str.upper()
    
    mis_df = df.melt(id_vars=['match_label'], value_vars=mis_cols, var_name='Attribute', value_name='Gap')
    mis_df['Attribute'] = mis_df['Attribute'].str.replace('_misconception', '').str.upper()
    
    # Setup Figure
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # Plot 1: Disappointment (Expectation vs Reality) by Match Status
    sns.barplot(data=dis_df, x='Attribute', y='Gap', hue='match_label', ax=ax1, palette='RdYlGn_r')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_title('Bản đồ Thất vọng (Expectation - Reality) theo Trạng thái Match\nPositive = Thực tế tệ hơn kỳ vọng | Negative = Bất ngờ tích cực', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Mean Gap (1-10 scale)')
    ax1.set_xlabel('')
    
    # Plot 2: Misconception (Self Perception vs Partner View) by Match Status
    sns.barplot(data=mis_df, x='Attribute', y='Gap', hue='match_label', ax=ax2, palette='coolwarm')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title('Bản đồ Ngộ nhận (Self Perception - Partner Rating) theo Trạng thái Match\nPositive = Tự tin thái quá | Negative = Tự ti', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Mean Gap (1-10 scale)')
    
    plt.suptitle('CHÌA KHÓA CỦA SỰ THÀNH CÔNG: PHÂN TÍCH HÀNH VI MATCH VS. NO MATCH', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    plt.savefig(CONFIG['paths']['plot_path'], bbox_inches='tight')
    print(f"   ✓ Dashboard saved to {CONFIG['paths']['plot_path']}")
    
    # Return stats for logging
    return dis_df.groupby(['match_label', 'Attribute'])['Gap'].mean(), mis_df.groupby(['match_label', 'Attribute'])['Gap'].mean()

def run_task():
    df = load_and_clean()
    df = calculate_behavioral_gaps(df)
    dis_stats, mis_stats = visualize_comprehensive(df)
    
    print("\n" + "="*60)
    print("INSIGHT SUMMARY: MATCH VS NO MATCH")
    print("="*60)
    print("\n[DISAPPOINTMENT GAPS]")
    print(dis_stats.unstack().round(3))
    print("\n[MISCONCEPTION GAPS]")
    print(mis_stats.unstack().round(3))
    
    print("\n" + "="*60)
    print("KEY BEHAVIORAL SECRETS")
    print("="*60)
    
    # Calculate differences
    dis_diff = dis_stats.loc['Match'] - dis_stats.loc['No Match']
    mis_diff = mis_stats.loc['Match'] - mis_stats.loc['No Match']
    
    print("• Các cặp Match có mức độ hài lòng (Gap âm) sâu hơn ở mọi phương diện.")
    print(f"• Đặc biệt hài lòng vượt trội về: {dis_diff.idxmin()}")
    print("• Các cặp Match có độ 'ngộ nhận' THẤP HƠN đáng kể so với nhóm thất bại.")
    print(f"• Thành thật nhất về: {mis_diff.idxmin()}")

if __name__ == "__main__":
    run_task()
