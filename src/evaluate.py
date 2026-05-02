"""
Evaluation

Loads the zero-shot modeling results and generates statistical summaries 
and visualizations for the GMM and OC-SVM anomaly detection pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Define paths
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "data" / "features" / "zero_shot_results.csv"
RESPONSES_CSV = REPO_ROOT / "data" / "features" / "responses_features.csv"

def run_evaluation():
    print(f"Loading results from: {RESULTS_CSV}...")
    try:
        df = pd.read_csv(RESULTS_CSV)
    except FileNotFoundError:
        print("Error: Could not find results CSV. Have you run the modeling pipeline yet?")
        return

    # LOAD & MERGE PSYCHOLOGICAL RESPONSES
    print("Loading psychological responses...")
    try:
        resp_df = pd.read_csv(RESPONSES_CSV)
        # We only care about how they felt during the puzzle (Phase 2)
        p2_df = resp_df[resp_df['phase'] == 'phase2']
        emotions = ['frustrated', 'difficulty', 'nervous']
        # Group by subject and average across all rounds
        grouped_resp = p2_df.groupby('subject_id')[emotions].mean().reset_index()
        
        # Merge with main dataframe
        df = df.merge(grouped_resp, on='subject_id', how='left')
        has_psych_data = True
    except FileNotFoundError:
        print("Warning: responses_features.csv not found. Skipping psychological correlation.")
        has_psych_data = False

    # ==========================================
    # METRICS & STATISTICS
    # ==========================================
    # Calculate Separation (Delta) for continuous scores
    df['GMM_Separation'] = df['GMM_P2_Stress'] - df['GMM_P1']
    df['SVM_Separation'] = df['SVM_P2_Stress'] - df['SVM_P1']

    print("\n" + "="*50)
    print("1. PHYSIOLOGICAL SEPARATION (STRESS JUMP)")
    print("="*50)
    print(f"GMM Mean Jump: {df['GMM_Separation'].mean():+.2f}")
    print(f"SVM Mean Jump: {df['SVM_Separation'].mean():+.2f}")

    if has_psych_data:
        print("\n" + "="*50)
        print("2. PSYCHOLOGICAL CORRELATION MATRIX")
        print("="*50)
        corr_cols = ['GMM_Separation', 'SVM_Separation', 'frustrated', 'difficulty', 'nervous']
        corr_matrix = df[corr_cols].corr()
        print(corr_matrix.loc[['GMM_Separation', 'SVM_Separation'], ['frustrated', 'difficulty', 'nervous']])

    # ==========================================
    # VISUALIZATIONS
    # ==========================================
    sns.set_theme(style="whitegrid")

    # PLOT 1: PERCENTAGE FLAGGING RATES
    print("Generating Plot 2: Percentage Flagging Rates...")
    plot_df_pct = df.melt(
        id_vars=['subject_id'], 
        value_vars=['SVM_Rate_P1', 'SVM_Rate_P2', 'SVM_Rate_P3', 'GMM_Rate_P1', 'GMM_Rate_P2', 'GMM_Rate_P3'],
        var_name='Condition', value_name='Flagged_Percentage'
    )
    plot_df_pct['Model'] = plot_df_pct['Condition'].apply(lambda x: x.split('_')[0])
    plot_df_pct['Phase'] = plot_df_pct['Condition'].apply(lambda x: 'P1 (Rest)' if 'P1' in x else ('P2 (Stress)' if 'P2' in x else 'P3 (Recovery)'))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df_pct, x='Phase', y='Flagged_Percentage', hue='Model', 
        palette={'GMM': '#E88854', 'SVM': '#9a0000'}, capsize=0.1, alpha=1.0
    )
    plt.title('Percentage of Windows Flagged as Anomalies', fontsize=16, pad=15)
    plt.ylabel('Percentage Flagged (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Experimental Phase', fontsize=16)
    plt.legend(title='Model', loc='upper left', frameon=True)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16)
    plt.legend(title='Model', loc='upper left', frameon=True, 
               fontsize=16, title_fontsize=16)
    plt.tight_layout()
    plt.savefig("report/figures/phase_anomalies.pdf")

    # PLOT 2: 2x3 PSYCHOLOGICAL CORRELATION GRID
    if has_psych_data:
        print("Generating Plot 3: 2x3 Psychological Correlation Grid...")
        fig3, axes = plt.subplots(2, 3, figsize=(16, 11)) 
        sns.set_theme(style="ticks")

        models = [('GMM_Separation', 'GMM', '#E88854', '#E86E54'), ('SVM_Separation', 'SVM', '#9a0000', '#E8A254')]
        emotions = ['frustrated', 'difficulty', 'nervous']

        for row_idx, (col_name, model_name, scatter_color, line_color) in enumerate(models):
            for col_idx, emotion in enumerate(emotions):
                ax = axes[row_idx, col_idx]
                
                # Plot the regression
                sns.regplot(data=df, x=col_name, y=emotion, ax=ax,
                            scatter_kws={'alpha': 0.6, 's': 60, 'color': scatter_color}, 
                            line_kws={'color': line_color, 'linewidth': 2})
                
                # Calculate r and R^2
                valid_data = df[[col_name, emotion]].dropna()
                if len(valid_data) > 1:
                    r_val = valid_data[col_name].corr(valid_data[emotion])
                    r_squared = r_val ** 2
                else:
                    r_val, r_squared = 0, 0
                
                # Format the title to include the stats on a second line
                title_text = f'{model_name} Detection vs. {emotion.capitalize()}\n$r = {r_val:+.2f}$  |  $R^2 = {r_squared:.2f}$'
                ax.set_title(title_text, fontsize=13, fontweight='bold', pad=10)
                
                ax.set_xlabel(f'{model_name} Anomaly Jump (P2 - P1)', fontsize=10)
                
                if emotion == 'difficulty':
                    ax.set_ylabel('Reported Difficulty (0-10)', fontsize=10)
                else:
                    ax.set_ylabel(f'Reported {emotion.capitalize()} (Delta)', fontsize=10)
                ax.grid(True, alpha=0.3)

        plt.suptitle('Physiological Anomaly vs. Psychological Reality', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig("report/figures/responses_correlation.pdf")

if __name__ == "__main__":
    run_evaluation()