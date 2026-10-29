"""
src/feature_selection.py
------------------------
Feature Selection & Exploratory Data Analysis (EDA) Module
Food Adulteration Risk Prediction Using Data Mining Techniques

Generates 8 publication-grade visualization figures saved to reports/figures/
and performs comparative multi-method feature selection:
1. Pearson Correlation Filter
2. Chi-Square Statistical Test (SelectKBest)
3. Random Forest Feature Importance (MDI)

CRITICAL DISCLAIMER:
SYNTHETIC DATA — generated for academic demonstration.
NOT a substitute for accredited laboratory testing.
"""

import os
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# Global styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

SYNTHETIC_LABEL = "SYNTHETIC DATA — generated for academic demonstration"
FIGURES_DIR = "reports/figures"


def generate_eda_figures(raw_csv_path: str = "data/raw/food_adulteration_raw.csv", output_dir: str = FIGURES_DIR) -> Dict[str, str]:
    """
    Generate at least 8 publication-quality EDA figures with academic annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_csv_path)
    figure_paths = {}

    # --------------------------------------------------------------------------
    # Fig 1: Class Balance
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['is_adulterated'].value_counts()
    colors = ['#2b5c8f', '#d9534f']
    bars = ax.bar(['Safe (0)', 'Adulterated (1)'], counts.values, color=colors, width=0.55, edgecolor='black', alpha=0.9)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}\n({height/len(df):.1%})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Figure 1: Target Class Balance (Safe vs Adulterated)\n' + SYNTHETIC_LABEL, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, max(counts.values) * 1.18)
    fig1_path = os.path.join(output_dir, 'fig1_class_balance.png')
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    figure_paths['fig1'] = fig1_path

    # --------------------------------------------------------------------------
    # Fig 2: pH Distribution by Class
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df[df['is_adulterated'] == 0]['ph_level'], ax=ax, label='Safe', fill=True, color='#2b5c8f', alpha=0.4, linewidth=2)
    sns.kdeplot(data=df[df['is_adulterated'] == 1]['ph_level'], ax=ax, label='Adulterated', fill=True, color='#d9534f', alpha=0.4, linewidth=2)
    ax.set_xlabel('pH Level (Physical Acidity / Alkalinity)')
    ax.set_ylabel('Empirical Density')
    ax.set_title('Figure 2: Probability Density of pH by Adulteration Class\n' + SYNTHETIC_LABEL, fontsize=12, fontweight='bold', pad=12)
    ax.legend(title='Sample Status', frameon=True)
    fig2_path = os.path.join(output_dir, 'fig2_ph_distribution_by_class.png')
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    figure_paths['fig2'] = fig2_path

    # --------------------------------------------------------------------------
    # Fig 3: Moisture Content by Food Category
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(x='food_category', y='moisture_content', hue='is_adulterated', data=df,
                palette={0: '#2b5c8f', 1: '#d9534f'}, ax=ax, fliersize=2.5)
    ax.set_xlabel('Food Category')
    ax.set_ylabel('Moisture Content (% w/w)')
    ax.set_title('Figure 3: Moisture Content Distribution across Food Matrix\n' + SYNTHETIC_LABEL, fontsize=12, fontweight='bold', pad=12)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Safe', 'Adulterated'], title='Status', frameon=True)
    fig3_path = os.path.join(output_dir, 'fig3_moisture_by_food_category.png')
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    figure_paths['fig3'] = fig3_path

    # --------------------------------------------------------------------------
    # Fig 4: Correlation Heatmap
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols = ['moisture_content', 'ph_level', 'storage_temperature', 'expiry_days_remaining',
                'color_score', 'smell_score', 'taste_score', 'packaging_quality_score', 'is_adulterated']
    corr_matrix = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1,
                mask=mask, ax=ax, square=True, cbar_kws={'shrink': 0.8}, linewidths=0.5)
    ax.set_title('Figure 4: Inter-Parameter Correlation Matrix Heatmap\n' + SYNTHETIC_LABEL, fontsize=12, fontweight='bold', pad=12)
    fig4_path = os.path.join(output_dir, 'fig4_correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    figure_paths['fig4'] = fig4_path

    # --------------------------------------------------------------------------
    # Fig 5: Sensory Score Boxplots
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    sensory_vars = [('color_score', 'Color Score (1-5)'),
                    ('smell_score', 'Smell Score (1-5)'),
                    ('taste_score', 'Taste Score (1-5)'),
                    ('packaging_quality_score', 'Packaging Integrity (1-5)')]

    for idx, (var, label) in enumerate(sensory_vars):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        sns.boxplot(x='is_adulterated', y=var, hue='is_adulterated', data=df, ax=ax,
                    palette={0: '#2b5c8f', 1: '#d9534f'}, width=0.45, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Safe', 'Adulterated'])
        ax.set_xlabel('')
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight='bold')

    fig.suptitle('Figure 5: Sensory and Packaging Degradation by Adulteration Class\n' + SYNTHETIC_LABEL,
                 fontsize=13, fontweight='bold', y=0.98)
    fig5_path = os.path.join(output_dir, 'fig5_sensory_score_boxplots.png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    figure_paths['fig5'] = fig5_path

    # --------------------------------------------------------------------------
    # Fig 6: Adulteration Rate by Food Category
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_ad_rate = df.groupby('food_category')['is_adulterated'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=cat_ad_rate.index, y=cat_ad_rate.values, hue=cat_ad_rate.index, palette='magma', ax=ax, edgecolor='black', alpha=0.9, legend=False)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3),
                    textcoords='offset points')
    ax.set_ylabel('Adulteration Prevalence Rate (%)')
    ax.set_xlabel('Food Category')
    ax.set_title('Figure 6: Adulteration Prevalence Rate across Food Categories\n' + SYNTHETIC_LABEL, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, max(cat_ad_rate.values) * 1.18)
    fig6_path = os.path.join(output_dir, 'fig6_adulteration_rate_by_category.png')
    plt.tight_layout()
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    figure_paths['fig6'] = fig6_path

    # --------------------------------------------------------------------------
    # Fig 7: Storage Temperature vs Risk
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(x='storage_temperature', y='ground_truth_risk_score', hue='risk_band',
                    data=df, palette={'Low Risk': '#28a745', 'Medium Risk': '#ffc107', 'High Risk': '#dc3545'},
                    alpha=0.6, ax=ax, s=35, edgecolor=None)
    ax.axhline(0.35, color='gray', linestyle='--', linewidth=1, label='Low/Medium Threshold (0.35)')
    ax.axhline(0.65, color='black', linestyle='--', linewidth=1, label='Medium/High Threshold (0.65)')
    ax.set_xlabel('Storage Temperature (°C)')
    ax.set_ylabel('Calibrated Continuous Risk Score')
    ax.set_title('Figure 7: Impact of Storage Temperature on Contamination Risk\n' + SYNTHETIC_LABEL, fontsize=12, fontweight='bold', pad=12)
    ax.legend(title='Operational Risk Band', frameon=True, loc='upper left')
    fig7_path = os.path.join(output_dir, 'fig7_storage_temperature_vs_risk.png')
    plt.tight_layout()
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    figure_paths['fig7'] = fig7_path

    # --------------------------------------------------------------------------
    # Fig 8: Expiry Days Remaining vs Risk
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x='expiry_days_remaining', hue='is_adulterated', multiple='stack',
                 palette={0: '#2b5c8f', 1: '#d9534f'}, bins=40, ax=ax, alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Expired (Day 0)')
    ax.set_xlabel('Shelf Life Days Remaining (Negative = Expired)')
    ax.set_ylabel('Sample Frequency')
    ax.set_title('Figure 8: Distribution of Expiry Days by Adulteration Status\n' + SYNTHETIC_LABEL, fontsize=12, fontweight='bold', pad=12)
    ax.legend(title='Status', labels=['Adulterated (1)', 'Safe (0)', 'Expired (Day 0)'], frameon=True)
    fig8_path = os.path.join(output_dir, 'fig8_expiry_vs_risk.png')
    plt.tight_layout()
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    figure_paths['fig8'] = fig8_path

    print(f"[EDA] Successfully generated and saved {len(figure_paths)} figures in {output_dir}")
    return figure_paths


def perform_tri_method_feature_selection(processed_train_csv: str = "data/processed/train_preprocessed.csv") -> pd.DataFrame:
    """
    Applies three distinct feature selection methodologies and produces a unified
    comparative ranking table:
    1. Pearson Correlation Filter (linear correlation magnitude with target)
    2. Chi-Square Test (univariate dependency of positive features on target)
    3. Random Forest Gini Importance (Mean Decrease in Impurity / MDI)
    """
    df = pd.read_csv(processed_train_csv)
    target = 'is_adulterated'
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # --------------------------------------------------------------------------
    # Method 1: Pearson Correlation Filter
    # --------------------------------------------------------------------------
    correlations = X.apply(lambda col: col.corr(y))
    corr_abs = correlations.abs()
    rank_corr = corr_abs.rank(ascending=False, method='min').astype(int)

    # --------------------------------------------------------------------------
    # Method 2: Chi-Square Test
    # (Features must be non-negative; scale X to [0, 1] range for chi2)
    # --------------------------------------------------------------------------
    scaler = MinMaxScaler()
    X_pos = scaler.fit_transform(X)
    chi2_scores, p_values = chi2(X_pos, y)
    rank_chi2 = pd.Series(chi2_scores, index=feature_cols).rank(ascending=False, method='min').astype(int)

    # --------------------------------------------------------------------------
    # Method 3: Random Forest Feature Importance (MDI)
    # --------------------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importances = pd.Series(rf.feature_importances_, index=feature_cols)
    rank_rf = rf_importances.rank(ascending=False, method='min').astype(int)

    # --------------------------------------------------------------------------
    # Unified Comparative Table
    # --------------------------------------------------------------------------
    comparison_df = pd.DataFrame({
        'Feature': feature_cols,
        'Pearson_Abs_Corr': np.round(corr_abs.values, 4),
        'Rank_Correlation': rank_corr.values,
        'Chi2_Statistic': np.round(chi2_scores, 2),
        'Chi2_p_value': [f"{p:.2e}" for p in p_values],
        'Rank_Chi2': rank_chi2.values,
        'RF_Gini_Importance': np.round(rf_importances.values, 4),
        'Rank_Random_Forest': rank_rf.values
    })

    # Compute Consensus Average Rank
    comparison_df['Average_Rank'] = np.round(
        (comparison_df['Rank_Correlation'] + comparison_df['Rank_Chi2'] + comparison_df['Rank_Random_Forest']) / 3.0, 2
    )
    comparison_df = comparison_df.sort_values('Average_Rank').reset_index(drop=True)

    print("\n=== TRI-METHOD FEATURE SELECTION COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))
    return comparison_df

if __name__ == "__main__":
    generate_eda_figures()
    perform_tri_method_feature_selection()
