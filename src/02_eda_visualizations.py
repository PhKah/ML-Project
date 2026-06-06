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
        'plot_dir': 'plots/eda/'
    }
}

if not os.path.exists(CONFIG['paths']['plot_dir']):
    os.makedirs(CONFIG['paths']['plot_dir'])

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    print(f"[1] Loading data...")
    return pd.read_csv(CONFIG['paths']['input_data'], encoding='ISO-8859-1')

def plot_class_imbalance(df):
    print(f"[2] Plotting Class Imbalance...")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='match', data=df, palette='viridis')
    plt.title('Tỷ lệ Match vs No Match (Class Imbalance)', fontsize=15, fontweight='bold')
    plt.xlabel('Trạng thái Match (0=No, 1=Yes)')
    plt.ylabel('Số lượng bản ghi')
    
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_height() + 50
        ax.annotate(percentage, (x, y), fontsize=12, fontweight='bold')
        
    plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '01_class_imbalance.png'))
    plt.close()

def plot_demographics(df):
    print(f"[3] Plotting Demographics...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(df['age'].dropna(), kde=True, color='skyblue', ax=ax1)
    ax1.set_title('Phân phối Tuổi của người tham gia', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tuổi')
    
    race_map = {1: 'Black', 2: 'White', 3: 'Hispanic', 4: 'Asian', 5: 'Native', 6: 'Other'}
    race_df = df['race'].map(race_map).value_counts().reset_index()
    sns.barplot(x='count', y='race', data=race_df, palette='magma', ax=ax2)
    ax2.set_title('Phân bổ Chủng tộc', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Số lượng')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '02_demographics.png'))
    plt.close()

def plot_attraction_determinants(df):
    print(f"[4] Plotting Attraction Determinants...")
    attrs = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
    corr = df[attrs + ['match']].corr()['match'].drop('match').sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr.values, y=corr.index, palette='RdYlGn')
    plt.title('Tương quan giữa 6 Thuộc tính cốt lõi và quyết định Match', fontsize=15, fontweight='bold')
    plt.xlabel('Hệ số tương quan (Correlation Coefficient)')
    plt.axvline(0, color='black', lw=1)
    
    plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '03_attraction_determinants.png'))
    plt.close()

def plot_choice_paradox(df):
    print(f"[5] Plotting H1: Choice Paradox...")
    round_stats = df.groupby('round')['match'].mean().reset_index()
    round_stats['match'] *= 100
    
    plt.figure(figsize=(10, 6))
    sns.regplot(x='round', y='match', data=round_stats, color='red', marker='o')
    plt.title('H1: Nghịch lý lựa chọn (Choice Paradox)\nTỷ lệ Match giảm khi số lượng người gặp mặt tăng', fontsize=15, fontweight='bold')
    plt.xlabel('Số lượng người gặp mặt trong một Wave (Round)')
    plt.ylabel('Tỷ lệ Match trung bình (%)')
    
    plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '04_choice_paradox.png'))
    plt.close()

def plot_homophily_interests(df):
    print(f"[6] Plotting H2: Homophily (Interest Similarity Deep Dive)...")
    if 'int_corr' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # 6a. KDE Plot (Existing)
        sns.kdeplot(data=df, x='int_corr', hue='match', fill=True, common_norm=False, palette='crest', ax=ax1)
        ax1.set_title('Phân phối Tương quan sở thích\nMatch vs No Match', fontsize=13, fontweight='bold')

        # 6b. Match Rate by Quartile (Deep Dive)
        df['int_corr_bin'] = pd.qcut(df['int_corr'], 4, labels=['Q1 (Thấp)', 'Q2', 'Q3', 'Q4 (Cao)'])
        match_rate = df.groupby('int_corr_bin')['match'].mean().reset_index()
        match_rate['match'] *= 100

        sns.barplot(x='int_corr_bin', y='match', data=match_rate, palette='viridis', ax=ax2)
        ax2.set_title('Tỷ lệ Match theo phân đoạn Tương quan sở thích\n(Higher Interest Corr -> Higher Success?)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Tỷ lệ Match (%)')
        ax2.set_xlabel('Tứ phân vị (Quartiles) của int_corr')

        # Add labels on bars
        for p in ax2.patches:
            ax2.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '05_homophily_interests_deep.png'))
        plt.close()

        # 6c. Interaction: Interest vs. Attractiveness (The Compensatory Effect)
        if 'attr' in df.columns:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df.sample(min(2000, len(df))), x='int_corr', y='attr', hue='match', 
                            alpha=0.6, palette={0: 'gray', 1: 'red'}, style='match', markers={0: 's', 1: 'o'})
            plt.title('H2 Deep Dive: Sự bù trừ giữa Tương quan sở thích và Ngoại hình', fontsize=15, fontweight='bold')
            plt.xlabel('Tương quan sở thích (int_corr)')
            plt.ylabel('Điểm Ngoại hình đối phương chấm (attr)')
            plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '05_homophily_interaction.png'))
            plt.close()


def plot_homophily_age(df):
    print(f"[7] Plotting H3: Age Proximity...")
    if 'age_o' in df.columns:
        df['age_gap'] = (df['age'] - df['age_o']).abs()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='match', y='age_gap', data=df, palette='Set2')
        plt.title('H3: Khoảng cách Tuổi tác vs Trạng thái Match', fontsize=15, fontweight='bold')
        plt.xlabel('Trạng thái Match (0=No, 1=Yes)')
        plt.ylabel('Khoảng cách tuổi (Age Gap)')
        
        plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '06_homophily_age.png'))
        plt.close()

def plot_ego_bias(df):
    print(f"[8] Plotting H4: Cognitive Humility (Ego Bias)...")
    attrs = ['attr', 'sinc', 'intel', 'fun', 'amb']
    bias_data = []
    for a in attrs:
        s31 = f'{a}3_1'
        s51 = f'{a}5_1'
        if s31 in df.columns and s51 in df.columns:
            bias = df[s31] - df[s51]
            temp_df = pd.DataFrame({'Attribute': a.upper(), 'Bias': bias, 'match': df['match']})
            bias_data.append(temp_df)
            
    if bias_data:
        full_bias_df = pd.concat(bias_data).reset_index(drop=True)
        plt.figure(figsize=(12, 7))
        sns.barplot(data=full_bias_df, x='Attribute', y='Bias', hue='match', palette='coolwarm')
        plt.axhline(0, color='black', lw=1)
        plt.title('H4: Sự ngộ nhận (Ego Bias)\nNhóm Match có xu hướng ít tự tin thái quá hơn', fontsize=15, fontweight='bold')
        plt.ylabel('Mức độ Ngộ nhận (3_1 - 5_1)')
        
        plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '07_ego_bias.png'))
        plt.close()

def plot_anti_leakage_audit(df):
    print(f"[9] Plotting Anti-Leakage Audit...")
    static_pref = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
    dynamic_ratings = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
    
    static_corr = df[static_pref + ['match']].corr()['match'].drop('match').abs().mean()
    dynamic_corr = df[dynamic_ratings + ['match']].corr()['match'].drop('match').abs().mean()
    
    audit_df = pd.DataFrame({
        'Feature Group': ['Static (Profile)', 'Dynamic (Post-Date)'],
        'Avg Correlation with Match': [static_corr, dynamic_corr]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature Group', y='Avg Correlation with Match', data=audit_df, palette='Set1')
    plt.title('Kiểm định Chống rò rỉ (Anti-Leakage Audit)\nBiến Động có tương quan ảo cực cao so với biến Tĩnh', fontsize=15, fontweight='bold')
    plt.ylabel('Hệ số tương quan trung bình (Absolute Mean)')
    
    plt.savefig(os.path.join(CONFIG['paths']['plot_dir'], '08_leakage_audit.png'))
    plt.close()

if __name__ == "__main__":
    df = load_data()
    plot_class_imbalance(df)
    plot_demographics(df)
    plot_attraction_determinants(df)
    plot_choice_paradox(df)
    plot_homophily_interests(df)
    plot_homophily_age(df)
    plot_ego_bias(df)
    plot_anti_leakage_audit(df)
    print(f"\n✓ All EDA & Audit plots generated in {CONFIG['paths']['plot_dir']}")
