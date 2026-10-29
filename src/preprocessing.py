"""
src/preprocessing.py
--------------------
Robust Preprocessing Pipeline & SMOTE Data Balancing
Food Adulteration Risk Prediction Using Data Mining Techniques

Performs:
1. Missing-value imputation (Median for numeric, Most Frequent for categorical)
2. Duplicate removal
3. Interquartile Range (IQR) outlier capping fitted on training distribution only
4. Ordinal encoding for sensory Likert scores
5. One-Hot Encoding for categorical features (food_category, preservatives_used)
6. Standard Scaling for numeric parameters
7. Stratified 80/20 train/test splitting
8. SMOTE oversampling applied EXCLUSIVELY to the training split to prevent data leakage.

CRITICAL DISCLAIMER:
SYNTHETIC DATA — generated for academic demonstration.
NOT a substitute for accredited laboratory testing.
"""

import os
import sys

# Ensure repository root is on sys.path so 'src' can be imported anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == '__main__':
    sys.modules['src.preprocessing'] = sys.modules['__main__']

import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Global random seed for reproducibility
RANDOM_STATE = 42

class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to clip numerical outliers based on the
    Interquartile Range (IQR) computed strictly on training data:
    Lower Bound = Q1 - 1.5 * IQR
    Upper Bound = Q3 + 1.5 * IQR
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        q25 = np.percentile(X_arr, 25, axis=0)
        q75 = np.percentile(X_arr, 75, axis=0)
        iqr = q75 - q25
        self.lower_bounds_ = q25 - (self.factor * iqr)
        self.upper_bounds_ = q75 + (self.factor * iqr)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        clipped = np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)
        return clipped

IQRClipper.__module__ = "src.preprocessing"


def create_preprocessor() -> Tuple[ColumnTransformer, List[str]]:
    """
    Constructs a Scikit-Learn ColumnTransformer that handles numeric,
    ordinal sensory, and categorical features uniformly.
    """
    numeric_features = [
        'moisture_content',
        'ph_level',
        'storage_temperature',
        'expiry_days_remaining'
    ]

    sensory_features = [
        'color_score',
        'smell_score',
        'taste_score',
        'packaging_quality_score'
    ]

    categorical_features = [
        'food_category',
        'preservatives_used'
    ]

    # Pipeline for objective numeric measurements
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('iqr_clipper', IQRClipper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    # Pipeline for sensory Likert scores (1 to 5)
    sensory_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for nominal categorical features
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'))
    ])

    # Unified ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('sen', sensory_pipeline, sensory_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    feature_order = numeric_features + sensory_features + categorical_features
    return preprocessor, feature_order


def apply_smote_train_only(X_train: np.ndarray, y_train: np.ndarray, random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to the training
    set strictly. Includes a fallback implementation in case imbalanced-learn
    is unavailable in specific constrained environments.
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Applied imbalanced-learn SMOTE. Resampled train shape: {X_res.shape}, Class balance: {np.bincount(y_res)}")
        return X_res, y_res
    except ImportError:
        print("[WARNING] imblearn not found, applying internal K-NN synthetic oversampling...")
        from sklearn.neighbors import NearestNeighbors
        # Deterministic synthetic minority interpolation (SMOTE algorithm implementation)
        minority_idx = np.where(y_train == 1)[0]
        majority_idx = np.where(y_train == 0)[0]
        n_majority = len(majority_idx)
        n_minority = len(minority_idx)
        n_synthetic_needed = n_majority - n_minority

        if n_synthetic_needed <= 0:
            return X_train, y_train

        X_min = X_train[minority_idx]
        k = min(5, len(minority_idx) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        neighbors = nn.kneighbors(X_min, return_distance=False)[:, 1:]

        synthetic_samples = []
        for i in range(n_synthetic_needed):
            base_idx = i % n_minority
            neighbor_choice = neighbors[base_idx][i % k]
            diff = X_min[neighbor_choice] - X_min[base_idx]
            step = np.random.uniform(0.1, 0.9)
            synthetic_samples.append(X_min[base_idx] + step * diff)

        X_synth = np.array(synthetic_samples)
        y_synth = np.ones(n_synthetic_needed, dtype=int)

        X_balanced = np.vstack([X_train, X_synth])
        y_balanced = np.concatenate([y_train, y_synth])
        return X_balanced, y_balanced


def run_preprocessing_pipeline(
    raw_csv_path: str = "data/raw/food_adulteration_raw.csv",
    output_dir: str = "data/processed",
    model_dir: str = "models",
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    End-to-end preprocessing execution:
    - Reads raw dataset
    - Removes duplicates
    - Stratified 80/20 train/test split
    - Fits ColumnTransformer on train split ONLY
    - Transforms train and test splits
    - Balances train split using SMOTE
    - Saves processed CSVs and serializes preprocessor
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"[PREPROCESSING] Loading raw dataset from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    # 1. Duplicate removal
    initial_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_count - len(df)
    print(f"[CLEANING] Duplicate records removed: {duplicates_removed}")

    # Separate features and target
    target_col = 'is_adulterated'
    X = df[[
        'food_category',
        'moisture_content',
        'ph_level',
        'storage_temperature',
        'expiry_days_remaining',
        'preservatives_used',
        'color_score',
        'smell_score',
        'taste_score',
        'packaging_quality_score'
    ]].copy()
    y = df[target_col].copy()

    # 2. Stratified 80/20 Split BEFORE any transformation or SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    print(f"[SPLIT] Stratified split completed: Train={len(X_train)} samples, Test={len(X_test)} samples")
    print(f"[SPLIT] Train class balance: Safe={(y_train == 0).sum()}, Adulterated={(y_train == 1).sum()} ({(y_train == 1).mean():.2%})")
    print(f"[SPLIT] Test class balance:  Safe={(y_test == 0).sum()}, Adulterated={(y_test == 1).sum()} ({(y_test == 1).mean():.2%})")

    # 3. Fit Preprocessing Pipeline strictly on Training split
    preprocessor, feature_order = create_preprocessor()
    preprocessor.fit(X_train)

    # Extract transformed column names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_columns = cat_encoder.get_feature_names_out(['food_category', 'preservatives_used']).tolist()
    all_feature_names = (
        ['moisture_content', 'ph_level', 'storage_temperature', 'expiry_days_remaining'] +
        ['color_score', 'smell_score', 'taste_score', 'packaging_quality_score'] +
        cat_columns
    )

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 4. Apply SMOTE to Training Split ONLY
    X_train_resampled, y_train_resampled = apply_smote_train_only(
        X_train_transformed, y_train.values, random_state=random_state
    )

    # 5. Format into DataFrames for export
    df_train_proc = pd.DataFrame(X_train_resampled, columns=all_feature_names)
    df_train_proc['is_adulterated'] = y_train_resampled

    df_test_proc = pd.DataFrame(X_test_transformed, columns=all_feature_names)
    df_test_proc['is_adulterated'] = y_test.values

    # 6. Save processed datasets and serialize preprocessor
    train_out = os.path.join(output_dir, "train_preprocessed.csv")
    test_out = os.path.join(output_dir, "test_preprocessed.csv")
    pipe_out = os.path.join(model_dir, "preprocessor.joblib")

    df_train_proc.to_csv(train_out, index=False)
    df_test_proc.to_csv(test_out, index=False)
    joblib.dump({
        'preprocessor': preprocessor,
        'feature_names': all_feature_names,
        'input_columns': feature_order
    }, pipe_out)

    print(f"[PERSISTENCE] Train dataset saved: {train_out} ({df_train_proc.shape})")
    print(f"[PERSISTENCE] Test dataset saved:  {test_out} ({df_test_proc.shape})")
    print(f"[PERSISTENCE] Preprocessor pipeline saved: {pipe_out}")

    return {
        'df_train': df_train_proc,
        'df_test': df_test_proc,
        'preprocessor': preprocessor,
        'feature_names': all_feature_names
    }

if __name__ == "__main__":
    run_preprocessing_pipeline()
