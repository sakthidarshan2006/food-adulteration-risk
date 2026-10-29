"""
src/evaluation.py
-----------------
Model Evaluation, Diagnostic Visualization & Pipeline Serialization Module
Food Adulteration Risk Prediction Using Data Mining Techniques

Evaluates all candidate classifiers on the hold-out test set:
1. Performance Metrics: Accuracy, Precision, Recall (Sensitivity), F1-Score, ROC-AUC
2. 5-Fold Cross-Validation (Mean ± Std)
3. Diagnostic Visualizations:
   - Overlaid ROC Curves (Figure 10)
   - 7-Model Confusion Matrix Grid (Figure 11)
   - Feature Importance & SHAP / Explainability Plot (Figure 12)
4. Safety-First Model Selection:
   - Prioritizes RECALL for the adulterated class (eliminates fatal false negatives)
5. Serializes the Winning Pipeline into models/best_model_pipeline.joblib

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

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.preprocessing import IQRClipper

RANDOM_STATE = 42
SYNTHETIC_LABEL = "SYNTHETIC DATA — generated for academic demonstration"
FIGURES_DIR = "reports/figures"
MODELS_DIR = "models"


def get_tuned_estimators() -> Dict[str, Any]:
    """
    Returns configured models with the optimal hyperparameters identified
    during the 5-fold GridSearchCV tuning phase.
    """
    return {
        'Decision Tree': DecisionTreeClassifier(
            criterion='entropy', max_depth=5, min_samples_split=2, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(var_smoothing=1e-7),
        'Support Vector Machine': SVC(
            C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        )
    }


def evaluate_all_models(
    train_csv: str = "data/processed/train_preprocessed.csv",
    test_csv: str = "data/processed/test_preprocessed.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Fits and evaluates all 7 candidate models on training and hold-out test sets.
    Computes 5-fold CV statistics and test metrics.
    """
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    X_train = df_train.drop(columns=['is_adulterated']).values
    y_train = df_train['is_adulterated'].values
    X_test = df_test.drop(columns=['is_adulterated']).values
    y_test = df_test['is_adulterated'].values

    estimators = get_tuned_estimators()
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics_list = []
    fitted_models = {}
    test_predictions = {}

    print(f"[EVALUATION] Benchmarking {len(estimators)} models on hold-out test set ({len(X_test)} samples)...")

    for name, clf in estimators.items():
        # Fit on full balanced training set
        clf.fit(X_train, y_train)
        fitted_models[name] = clf

        # 5-fold Cross-Validation Recall (Mean ± Std)
        cv_recalls = cross_val_score(clf, X_train, y_train, cv=cv_splitter, scoring='recall', n_jobs=1)
        cv_mean = float(np.mean(cv_recalls))
        cv_std = float(np.std(cv_recalls))

        # Predict on hold-out test set
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = clf.decision_function(X_test)

        test_predictions[name] = {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'cm': confusion_matrix(y_test, y_pred)
        }

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        metrics_list.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall (Adulterated)': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'CV_Recall_Mean': round(cv_mean, 4),
            'CV_Recall_Std': round(cv_std, 4),
            'CV_Recall_Formatted': f"{cv_mean:.4f} +/- {cv_std:.4f}"
        })

    metrics_df = pd.DataFrame(metrics_list)
    # Sort models by Recall descending (safety-first criteria)
    metrics_df = metrics_df.sort_values(by=['Recall (Adulterated)', 'ROC-AUC', 'F1-Score'], ascending=False).reset_index(drop=True)

    print("\n=== MODEL PERFORMANCE COMPARISON TABLE (SORTED BY RECALL) ===")
    print(metrics_df[['Model', 'Accuracy', 'Precision', 'Recall (Adulterated)', 'F1-Score', 'ROC-AUC', 'CV_Recall_Formatted']].to_string(index=False))

    return metrics_df, fitted_models, test_predictions


def generate_evaluation_figures(
    metrics_df: pd.DataFrame,
    test_predictions: Dict[str, Any],
    y_test: np.ndarray,
    fitted_models: Dict[str, Any],
    feature_names: List[str],
    output_dir: str = FIGURES_DIR
) -> Dict[str, str]:
    """
    Generates and saves Figures 10, 11, and 12 in reports/figures/:
    - Overlaid ROC Curves
    - Confusion Matrix Multi-Panel Grid
    - Feature Importance & Interpretability Plot
    """
    os.makedirs(output_dir, exist_ok=True)
    fig_paths = {}

    # --------------------------------------------------------------------------
    # Figure 10: Overlaid ROC Curves
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6.5))
    palette = sns.color_palette("tab10", len(test_predictions))

    for idx, (name, preds) in enumerate(test_predictions.items()):
        fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
        auc = roc_auc_score(y_test, preds['y_proba'])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2.0, color=palette[idx])

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.2, label='Chance Line (AUC = 0.50)')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 10: Comparative Receiver Operating Characteristic (ROC) Curves\n' + SYNTHETIC_LABEL,
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig10_path = os.path.join(output_dir, "fig10_model_comparison_roc.png")
    plt.tight_layout()
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    fig_paths['fig10'] = fig10_path

    # --------------------------------------------------------------------------
    # Figure 11: Confusion Matrix Multi-Panel Grid
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    model_names = list(test_predictions.keys())
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = test_predictions[name]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Safe', 'Adulterated'], yticklabels=['Safe', 'Adulterated'])
        ax.set_title(f"{name}\nRecall: {recall_score(y_test, test_predictions[name]['y_pred']):.2%}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Ground Truth')

    # Hide unused 8th subplot
    axes[7].axis('off')
    fig.suptitle('Figure 11: Confusion Matrix Comparison across All 7 Classifiers\n' + SYNTHETIC_LABEL,
                 fontsize=13, fontweight='bold', y=0.98)
    fig11_path = os.path.join(output_dir, "fig11_confusion_matrices.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    fig_paths['fig11'] = fig11_path

    # --------------------------------------------------------------------------
    # Figure 12: Feature Importance / Attribution Plot
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    rf_model = fitted_models.get('Random Forest')
    if rf_model is not None:
        importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(importances)))
        bars = ax.barh(importances.index, importances.values, color=colors, edgecolor='black', alpha=0.85)
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9, fontweight='bold')
        ax.set_xlabel('Mean Decrease in Impurity (Gini Importance)', fontsize=11, fontweight='bold')
        ax.set_title('Figure 12: Global Feature Importance Ranking (Random Forest Ensemble)\n' + SYNTHETIC_LABEL,
                     fontsize=12, fontweight='bold', pad=12)
        ax.set_xlim(0, max(importances.values) * 1.18)
    fig12_path = os.path.join(output_dir, "fig12_shap_feature_importance.png")
    plt.tight_layout()
    plt.savefig(fig12_path, dpi=300)
    plt.close()
    fig_paths['fig12'] = fig12_path

    print(f"[EVALUATION] Diagnostic figures generated: {list(fig_paths.keys())}")
    return fig_paths


def select_and_serialize_winning_model(
    metrics_df: pd.DataFrame,
    fitted_models: Dict[str, Any],
    preprocessor_joblib_path: str = "models/preprocessor.joblib",
    output_pipeline_path: str = "models/best_model_pipeline.joblib",
    output_metadata_path: str = "models/model_metadata.json"
) -> Dict[str, Any]:
    """
    Selects the optimal model based on Recall on the adulterated class,
    constructs the end-to-end inference Pipeline, and serializes artifacts.
    """
    # Safety-First Selection Criterion:
    # 1. Sort by Recall (Adulterated) descending
    # 2. Tie-break with ROC-AUC and F1-Score
    winner_row = metrics_df.sort_values(
        by=['Recall (Adulterated)', 'ROC-AUC', 'F1-Score'], ascending=False
    ).iloc[0]

    winning_model_name = winner_row['Model']
    winning_estimator = fitted_models[winning_model_name]

    print("\n" + "="*80)
    print(f"[WINNER] WINNING MODEL SELECTED: {winning_model_name}")
    print("="*80)
    print(f"Recall (Adulterated): {winner_row['Recall (Adulterated)']:.4f} ({winner_row['Recall (Adulterated)']:.2%})")
    print(f"ROC-AUC:              {winner_row['ROC-AUC']:.4f}")
    print(f"F1-Score:             {winner_row['F1-Score']:.4f}")
    print(f"Accuracy:             {winner_row['Accuracy']:.4f}")
    print(f"5-Fold CV Recall:     {winner_row['CV_Recall_Formatted']}")
    print("\n[JUSTIFICATION FOR RECALL SELECTION]")
    print(
        "In food safety inspection, a False Negative represents contaminated or adulterated food "
        "being erroneously certified as safe, causing severe toxic outbreaks and public health hazards. "
        "Conversely, a False Positive merely flags conforming food for confirmatory secondary laboratory "
        "testing. Therefore, maximizing Recall for the positive class is mathematically and ethically imperative."
    )

    # Load preprocessor to bundle into end-to-end Pipeline
    prep_data = joblib.load(preprocessor_joblib_path)
    preprocessor = prep_data['preprocessor']
    feature_names = prep_data['feature_names']
    input_columns = prep_data['input_columns']

    # Unified end-to-end Scikit-Learn Pipeline
    end_to_end_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', winning_estimator)
    ])

    os.makedirs(os.path.dirname(output_pipeline_path), exist_ok=True)
    joblib.dump(end_to_end_pipeline, output_pipeline_path)
    print(f"\n[SERIALIZATION] Complete end-to-end pipeline saved: {output_pipeline_path}")

    # Metadata record
    metadata = {
        'model_name': winning_model_name,
        'selection_metric': 'Recall (Adulterated Class)',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': float(winner_row['Accuracy']),
            'precision': float(winner_row['Precision']),
            'recall': float(winner_row['Recall (Adulterated)']),
            'f1_score': float(winner_row['F1-Score']),
            'roc_auc': float(winner_row['ROC-AUC']),
            'cv_recall_mean': float(winner_row['CV_Recall_Mean']),
            'cv_recall_std': float(winner_row['CV_Recall_Std'])
        },
        'input_columns': input_columns,
        'feature_names': feature_names,
        'risk_band_thresholds': {
            'low_risk': '< 0.35',
            'medium_risk': '0.35 - 0.65',
            'high_risk': '> 0.65'
        }
    }

    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"[SERIALIZATION] Metadata saved: {output_metadata_path}")

    return metadata


def run_evaluation_pipeline() -> Dict[str, Any]:
    """Runs Stage 5 end-to-end."""
    train_csv = "data/processed/train_preprocessed.csv"
    test_csv = "data/processed/test_preprocessed.csv"

    df_test = pd.read_csv(test_csv)
    y_test = df_test['is_adulterated'].values

    # 1. Benchmark models
    metrics_df, fitted_models, test_preds = evaluate_all_models(train_csv=train_csv, test_csv=test_csv)

    # 2. Generate Figures 10, 11, 12
    prep_data = joblib.load("models/preprocessor.joblib")
    feature_names = prep_data['feature_names']
    generate_evaluation_figures(metrics_df, test_preds, y_test, fitted_models, feature_names)

    # 3. Select and serialize winning pipeline
    metadata = select_and_serialize_winning_model(metrics_df, fitted_models)

    return {
        'metrics_df': metrics_df,
        'metadata': metadata
    }

if __name__ == "__main__":
    run_evaluation_pipeline()
