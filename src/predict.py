"""
src/predict.py
--------------
Prediction Engine & Risk Stratification Module
Architecture Stage 6: Food Adulteration Risk Prediction

Exposes:
- predict_single(data_dict): Predicts a single sample dictionary
- predict_batch(df): Predicts a batch DataFrame of food samples
- Returns: predicted_label, predicted_status, probability, risk_band, explanation
- Writes audit log to db/predictions.db via src/database.py

Calibrated Risk Band Definitions:
- < 0.35:      LOW RISK (Safe)
- 0.35 - 0.65: MEDIUM RISK (Retest advised)
- > 0.65:      HIGH RISK (Adulterated - lab testing recommended)

CRITICAL DISCLAIMER:
SYNTHETIC DATA — generated for academic demonstration.
NOT a substitute for accredited laboratory testing.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, List

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import IQRClipper
from src.database import log_prediction, log_predictions_batch

DEFAULT_PIPELINE_PATH = "models/best_model_pipeline.joblib"
DEFAULT_METADATA_PATH = "models/model_metadata.json"

# In-memory cached pipeline
_CACHED_PIPELINE = None
_CACHED_METADATA = None


def load_model_pipeline(pipeline_path: str = DEFAULT_PIPELINE_PATH, metadata_path: str = DEFAULT_METADATA_PATH):
    """Loads and caches the serialized winning end-to-end Pipeline and metadata."""
    global _CACHED_PIPELINE, _CACHED_METADATA
    if _CACHED_PIPELINE is None:
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Model pipeline not found at: {pipeline_path}. Run src/evaluation.py first.")
        _CACHED_PIPELINE = joblib.load(pipeline_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                _CACHED_METADATA = json.load(f)
        else:
            _CACHED_METADATA = {}

    return _CACHED_PIPELINE, _CACHED_METADATA


def determine_risk_band(probability: float) -> str:
    """
    Stratifies continuous predicted probability into operational risk bands:
    - < 0.35:      Low Risk (Safe)
    - 0.35 - 0.65: Medium Risk (Retest advised)
    - > 0.65:      High Risk (Adulterated - lab testing recommended)
    """
    if probability < 0.35:
        return "Low Risk"
    elif probability <= 0.65:
        return "Medium Risk"
    else:
        return "High Risk"


def compute_feature_attribution_single(sample_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Computes inspection risk driver factors for a single sample
    to explain WHY the model arrived at its verdict.
    """
    drivers = []
    cat = str(sample_dict.get('food_category', ''))
    m = float(sample_dict.get('moisture_content', 0.0))
    p = float(sample_dict.get('ph_level', 7.0))
    t = float(sample_dict.get('storage_temperature', 20.0))
    exp = int(sample_dict.get('expiry_days_remaining', 30))
    pres = str(sample_dict.get('preservatives_used', 'none'))

    c_score = int(sample_dict.get('color_score', 5))
    s_score = int(sample_dict.get('smell_score', 5))
    t_score = int(sample_dict.get('taste_score', 5))
    pkg_score = int(sample_dict.get('packaging_quality_score', 5))

    # Preservative rule
    if pres == 'non-permitted':
        drivers.append({'feature': 'Preservative', 'status': 'Violated', 'severity': 'High', 'detail': 'Unapproved chemical additive detected.'})
    # Expiry rule
    if exp <= 0:
        drivers.append({'feature': 'Expiry Status', 'status': 'Violated', 'severity': 'High', 'detail': f'Sample is already expired by {abs(exp)} days.'})
    elif exp <= 5:
        drivers.append({'feature': 'Shelf-Life', 'status': 'Warning', 'severity': 'Medium', 'detail': f'Near expiration ({exp} days remaining).'})

    # Sensory degradation
    if t_score <= 2:
        drivers.append({'feature': 'Taste Score', 'status': 'Violated', 'severity': 'High', 'detail': f'Abnormal / off-taste (Score {t_score}/5).'})
    if s_score <= 2:
        drivers.append({'feature': 'Smell Score', 'status': 'Violated', 'severity': 'High', 'detail': f'Rancid / foul odor (Score {s_score}/5).'})
    if c_score <= 2:
        drivers.append({'feature': 'Color Score', 'status': 'Violated', 'severity': 'High', 'detail': f'Unnatural tint / dye discoloration (Score {c_score}/5).'})
    if pkg_score <= 2:
        drivers.append({'feature': 'Packaging Quality', 'status': 'Violated', 'severity': 'Medium', 'detail': f'Damaged or unsealed packaging (Score {pkg_score}/5).'})

    # Physicochemical threshold breach
    if cat == 'milk':
        if m > 90.0:
            drivers.append({'feature': 'Milk Moisture', 'status': 'Violated', 'severity': 'High', 'detail': f'Excess water dilution ({m}% > 89%).'})
        if p > 7.2:
            drivers.append({'feature': 'Milk pH', 'status': 'Violated', 'severity': 'High', 'detail': f'Alkaline shift ({p} > 6.8) suggests neutralizer/detergent.'})
        elif p < 6.2:
            drivers.append({'feature': 'Milk pH', 'status': 'Violated', 'severity': 'High', 'detail': f'Acidic curdling ({p} < 6.5) suggests microbial spoilage.'})
        if t > 8.0:
            drivers.append({'feature': 'Cold Chain', 'status': 'Violated', 'severity': 'Medium', 'detail': f'Stored at {t}°C (Safe limit <= 6°C).'})
    elif cat == 'honey':
        if m > 20.0:
            drivers.append({'feature': 'Honey Moisture', 'status': 'Violated', 'severity': 'High', 'detail': f'Exceeds FSSAI maximum limit ({m}% > 20.0%).'})
    elif cat == 'oil':
        if m > 0.25:
            drivers.append({'feature': 'Oil Moisture', 'status': 'Violated', 'severity': 'High', 'detail': f'Moisture content {m}% exceeds pure oil limit (0.2%).'})

    if not drivers:
        drivers.append({'feature': 'All Standard Parameters', 'status': 'Conforming', 'severity': 'Low', 'detail': 'All physical, chemical, sensory, and logistical parameters comply with FSSAI norms.'})

    return drivers


def predict_single(
    data_dict: Dict[str, Any],
    log_to_db: bool = True,
    pipeline_path: str = DEFAULT_PIPELINE_PATH
) -> Dict[str, Any]:
    """
    Executes inference for a single food sample inspection.

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Dictionary of 10 food quality features.
    log_to_db : bool, default=True
        Whether to log the transaction to SQLite db/predictions.db.

    Returns
    -------
    Dict[str, Any]
        Inference dictionary containing predicted_label, predicted_status,
        probability, risk_band, and explainability risk drivers.
    """
    pipeline, metadata = load_model_pipeline(pipeline_path)

    # Convert dictionary into a single-row DataFrame
    required_cols = [
        'food_category', 'moisture_content', 'ph_level',
        'storage_temperature', 'expiry_days_remaining', 'preservatives_used',
        'color_score', 'smell_score', 'taste_score', 'packaging_quality_score'
    ]
    sample_df = pd.DataFrame([data_dict])[required_cols]

    # Predict label and continuous risk probability
    pred_label = int(pipeline.predict(sample_df)[0])
    if hasattr(pipeline, "predict_proba"):
        prob = float(pipeline.predict_proba(sample_df)[0][1])
    else:
        prob = float(pipeline.decision_function(sample_df)[0])

    status = "Adulterated" if pred_label == 1 else "Safe"
    risk_band = determine_risk_band(prob)
    drivers = compute_feature_attribution_single(data_dict)

    result = {
        'predicted_label': pred_label,
        'predicted_status': status,
        'probability': round(prob, 4),
        'risk_band': risk_band,
        'risk_drivers': drivers,
        'model_name': metadata.get('model_name', 'Gradient Boosting')
    }

    if log_to_db:
        log_id = log_prediction(data_dict, result)
        result['database_log_id'] = log_id

    return result


def predict_batch(
    df: pd.DataFrame,
    log_to_db: bool = True,
    pipeline_path: str = DEFAULT_PIPELINE_PATH
) -> pd.DataFrame:
    """
    Scans and scores an entire batch DataFrame of food samples.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least the 10 required features.
    log_to_db : bool, default=True
        Whether to bulk-log scored samples to SQLite db/predictions.db.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with predicted_label, predicted_status,
        probability, and risk_band.
    """
    pipeline, _ = load_model_pipeline(pipeline_path)

    required_cols = [
        'food_category', 'moisture_content', 'ph_level',
        'storage_temperature', 'expiry_days_remaining', 'preservatives_used',
        'color_score', 'smell_score', 'taste_score', 'packaging_quality_score'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input batch is missing required columns: {missing_cols}")

    X_batch = df[required_cols].copy()
    pred_labels = pipeline.predict(X_batch)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_batch)[:, 1]
    else:
        probs = pipeline.decision_function(X_batch)

    df_out = df.copy()
    df_out['predicted_label'] = pred_labels
    df_out['predicted_status'] = np.where(pred_labels == 1, 'Adulterated', 'Safe')
    df_out['probability'] = np.round(probs, 4)
    df_out['risk_band'] = [determine_risk_band(p) for p in probs]

    if log_to_db:
        log_predictions_batch(df_out)

    return df_out

if __name__ == "__main__":
    # Test sample inspection
    sample = {
        'food_category': 'milk',
        'moisture_content': 92.5,
        'ph_level': 7.8,
        'storage_temperature': 14.0,
        'expiry_days_remaining': -2,
        'preservatives_used': 'non-permitted',
        'color_score': 2,
        'smell_score': 2,
        'taste_score': 1,
        'packaging_quality_score': 2
    }
    res = predict_single(sample)
    print("\n=== SAMPLE PREDICTION INFERENCE RESULT ===")
    print(json.dumps(res, indent=2))
