"""
src/models.py
-------------
Multi-Paradigm Machine Learning & Data Mining Module
Food Adulteration Risk Prediction Using Data Mining Techniques

Contains:
1. Supervised Classification Benchmarking with 5-fold GridSearchCV:
   - Core Architecture: Decision Tree, Random Forest, Naive Bayes, Support Vector Machine (SVM)
   - Extensions: Logistic Regression, K-Nearest Neighbors (KNN), Gradient Boosting / XGBoost
2. Unsupervised Clustering:
   - K-Means with Elbow (Inertia) Method and Silhouette Score analysis
3. Association Rule Mining:
   - Apriori algorithm on domain-discretized parameters reporting top 10 rules by Lift

CRITICAL DISCLAIMER:
SYNTHETIC DATA — generated for academic demonstration.
NOT a substitute for accredited laboratory testing.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix

RANDOM_STATE = 42
SYNTHETIC_LABEL = "SYNTHETIC DATA — generated for academic demonstration"


def get_classification_models() -> Dict[str, Dict[str, Any]]:
    """
    Defines classification algorithms, their hyperparameter tuning grids,
    and academic rationale for food adulteration risk modeling.
    """
    models = {
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'max_depth': [3, 5, 8, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'rationale': "Provides transparent, rule-based if-then decision thresholds crucial for field inspectors to defend sample seizure legally."
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_leaf': [1, 2, 4]
            },
            'rationale': "Mitigates single-tree variance, captures complex non-linear feature interactions (e.g. pH threshold interacting with food category)."
        },
        'Naive Bayes': {
            'estimator': GaussianNB(),
            'param_grid': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            },
            'rationale': "Probabilistic baseline relying on Bayes' theorem; fast and effective for independent organoleptic indicator screening."
        },
        'Support Vector Machine': {
            'estimator': SVC(probability=True, random_state=RANDOM_STATE),
            'param_grid': {
                'C': [0.1, 1.0, 5.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'rationale': "Maximizes safety margin hyperplane in multidimensional feature space, robust against high-dimensional collinear sensory vectors."
        },
        'Logistic Regression': {
            'estimator': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'param_grid': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs']
            },
            'rationale': "Statistical benchmark producing well-calibrated odds ratios and direct sigmoid posterior risk probabilities."
        },
        'K-Nearest Neighbors': {
            'estimator': KNeighborsClassifier(n_jobs=-1),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'rationale': "Instance-based non-parametric classifier comparing new samples to past forensic library cases in feature space."
        },
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5]
            },
            'rationale': "Sequentially corrects residual errors, optimizing sensitive loss gradients for hard-to-classify borderline adulteration cases."
        }
    }
    return models


def train_and_tune_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = 'recall'
) -> Dict[str, Dict[str, Any]]:
    """
    Performs 5-fold Stratified GridSearchCV across all candidate algorithms.
    Optimizes for 'recall' to minimize dangerous false negatives in food safety.
    """
    model_definitions = get_classification_models()
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    trained_results = {}

    print(f"[MODEL TRAINING] Starting 5-fold GridSearchCV across {len(model_definitions)} algorithms (Scoring={scoring.upper()})...")

    for name, config in model_definitions.items():
        print(f"  -> Tuning {name}...")
        grid = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            scoring=scoring,
            cv=cv_splitter,
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train, y_train)

        trained_results[name] = {
            'best_estimator': grid.best_estimator_,
            'best_params': grid.best_params_,
            'best_cv_recall': grid.best_score_,
            'rationale': config['rationale']
        }
        print(f"     [DONE] {name} | Best CV Recall: {grid.best_score_:.4f} | Best Params: {grid.best_params_}")

    return trained_results


def evaluate_kmeans_clustering(
    X: np.ndarray,
    y_true: np.ndarray,
    k_range: range = range(2, 7),
    output_fig_path: str = "reports/figures/fig9_kmeans_elbow_silhouette.png"
) -> Dict[str, Any]:
    """
    Applies K-Means clustering across a range of k values to determine whether
    adulterated and safe samples form natural unsupervised geometric groupings.
    Computes Inertia (Elbow Method) and Silhouette Scores.
    """
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    inertias = []
    silhouettes = []
    k_values = list(k_range)

    print("[CLUSTERING] Evaluating K-Means clustering across k in", k_values)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_score = silhouette_score(X, cluster_labels, sample_size=min(2000, len(X)), random_state=RANDOM_STATE)
        silhouettes.append(sil_score)
        print(f"  -> k={k}: Inertia={km.inertia_:.1f}, Silhouette Score={sil_score:.4f}")

    # Plot Elbow and Silhouette curves
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = '#2b5c8f'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontsize=11, fontweight='bold')
    line1 = ax1.plot(k_values, inertias, marker='o', linewidth=2.5, color=color, label='Inertia (Elbow Curve)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color2 = '#d9534f'
    ax2.set_ylabel('Mean Silhouette Score', color=color2, fontsize=11, fontweight='bold')
    line2 = ax2.plot(k_values, silhouettes, marker='s', linewidth=2.5, linestyle='--', color=color2, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)
    plt.title('Figure 9: K-Means Clustering Validation (Elbow & Silhouette Curve)\n' + SYNTHETIC_LABEL, fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300)
    plt.close()

    # Optimal k=2 contingency matrix comparison
    km2 = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    labels_k2 = km2.fit_predict(X)
    ct = contingency_matrix(y_true, labels_k2)
    print(f"[CLUSTERING] Contingency Matrix for k=2 (True vs Cluster):\n{ct}")

    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'figure_path': output_fig_path,
        'contingency_k2': ct
    }


def run_apriori_association_mining(
    raw_csv_path: str = "data/raw/food_adulteration_raw.csv",
    min_support: float = 0.05,
    min_confidence: float = 0.60
) -> pd.DataFrame:
    """
    Applies Apriori Association Rule Mining on domain-binned quality indicators
    to discover high-lift decision patterns leading to adulteration.
    Works seamlessly with mlxtend or robust native frequent-itemset miner.
    """
    df = pd.read_csv(raw_csv_path)

    # Discretize continuous and ordinal attributes into domain transactional items
    transactions = pd.DataFrame()
    transactions['Adulterated'] = (df['is_adulterated'] == 1)
    transactions['NonPermitted_Preservative'] = (df['preservatives_used'] == 'non-permitted')
    transactions['Expired_Stock'] = (df['expiry_days_remaining'] <= 0)
    transactions['Near_Expiry'] = ((df['expiry_days_remaining'] > 0) & (df['expiry_days_remaining'] <= 10))
    transactions['Severe_Sensory_Defect'] = ((df['taste_score'] <= 2) | (df['smell_score'] <= 2) | (df['color_score'] <= 2))
    transactions['Packaging_Compromised'] = (df['packaging_quality_score'] <= 2)
    transactions['Storage_Temp_Abuse'] = (
        ((df['food_category'] == 'milk') & (df['storage_temperature'] > 8.0)) |
        ((df['food_category'] == 'beverages') & (df['storage_temperature'] > 15.0)) |
        (df['storage_temperature'] > 35.0)
    )
    transactions['Moisture_Anomalous'] = (
        ((df['food_category'] == 'milk') & (df['moisture_content'] > 90.0)) |
        ((df['food_category'] == 'honey') & (df['moisture_content'] > 20.0)) |
        ((df['food_category'] == 'oil') & (df['moisture_content'] > 0.25)) |
        ((df['food_category'] == 'spices') & (df['moisture_content'] > 13.0)) |
        ((df['food_category'] == 'grains') & (df['moisture_content'] > 14.5))
    )
    transactions['pH_Anomalous'] = (
        ((df['food_category'] == 'milk') & ((df['ph_level'] > 7.2) | (df['ph_level'] < 6.2))) |
        ((df['food_category'] == 'honey') & ((df['ph_level'] > 4.6) | (df['ph_level'] < 3.4))) |
        ((df['food_category'] == 'oil') & ((df['ph_level'] > 6.8) | (df['ph_level'] < 5.2)))
    )

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        # Filter for rules with consequent {'Adulterated'}
        adulterated_rules = rules[rules['consequents'] == frozenset({'Adulterated'})].copy()
        adulterated_rules = adulterated_rules.sort_values(by=['lift', 'confidence'], ascending=False).reset_index(drop=True)
        
        # Format for reporting
        formatted_rules = []
        for _, row in adulterated_rules.head(10).iterrows():
            formatted_rules.append({
                'Antecedents': " & ".join(list(row['antecedents'])),
                'Consequents': "Adulterated",
                'Support': round(float(row['support']), 4),
                'Confidence': round(float(row['confidence']), 4),
                'Lift': round(float(row['lift']), 4)
            })
        top_rules_df = pd.DataFrame(formatted_rules)

    except Exception as e:
        print(f"[NOTE] Using internal transactional association rule miner ({e})...")
        # Native deterministic implementation of Association Rule Mining
        total_n = len(transactions)
        p_adulterated = transactions['Adulterated'].mean()
        candidate_items = [c for c in transactions.columns if c != 'Adulterated']
        
        rules_list = []
        # Single-item antecedents
        for item in candidate_items:
            support_ant = transactions[item].mean()
            if support_ant >= min_support:
                support_both = (transactions[item] & transactions['Adulterated']).mean()
                conf = support_both / support_ant if support_ant > 0 else 0
                lift = conf / p_adulterated if p_adulterated > 0 else 0
                if conf >= min_confidence:
                    rules_list.append({
                        'Antecedents': item,
                        'Consequents': 'Adulterated',
                        'Support': round(support_both, 4),
                        'Confidence': round(conf, 4),
                        'Lift': round(lift, 4)
                    })
        # Two-item antecedents
        for i in range(len(candidate_items)):
            for j in range(i + 1, len(candidate_items)):
                item1, item2 = candidate_items[i], candidate_items[j]
                ant_mask = transactions[item1] & transactions[item2]
                support_ant = ant_mask.mean()
                if support_ant >= min_support:
                    support_both = (ant_mask & transactions['Adulterated']).mean()
                    conf = support_both / support_ant if support_ant > 0 else 0
                    lift = conf / p_adulterated if p_adulterated > 0 else 0
                    if conf >= min_confidence:
                        rules_list.append({
                            'Antecedents': f"{item1} & {item2}",
                            'Consequents': 'Adulterated',
                            'Support': round(support_both, 4),
                            'Confidence': round(conf, 4),
                            'Lift': round(lift, 4)
                        })

        top_rules_df = pd.DataFrame(rules_list).sort_values(by=['Lift', 'Confidence'], ascending=False).head(10).reset_index(drop=True)

    print("\n=== TOP 10 APRIORI ASSOCIATION RULES BY LIFT ===")
    print(top_rules_df.to_string(index=False))
    return top_rules_df


def run_model_building_pipeline(
    train_csv_path: str = "data/processed/train_preprocessed.csv",
    raw_csv_path: str = "data/raw/food_adulteration_raw.csv"
) -> Dict[str, Any]:
    """
    Executes Phase 5 end-to-end:
    - Trains and tunes 7 classification models using 5-fold CV
    - Evaluates K-Means unsupervised clustering
    - Discovers Apriori association rules
    """
    df_train = pd.read_csv(train_csv_path)
    X_train = df_train.drop(columns=['is_adulterated']).values
    y_train = df_train['is_adulterated'].values

    # 1. Supervised Modeling
    clf_results = train_and_tune_models(X_train, y_train, cv=5, scoring='recall')

    # 2. Unsupervised Clustering
    kmeans_results = evaluate_kmeans_clustering(X_train, y_train)

    # 3. Association Mining
    apriori_rules = run_apriori_association_mining(raw_csv_path=raw_csv_path)

    return {
        'classification_results': clf_results,
        'kmeans_results': kmeans_results,
        'apriori_rules': apriori_rules
    }

if __name__ == "__main__":
    run_model_building_pipeline()
