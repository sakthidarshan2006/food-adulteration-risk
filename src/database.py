"""
src/database.py
---------------
Persistent SQLite Database & Analytical Reporting Module
Architecture Stage 8: Food Adulteration Risk Prediction

Manages:
- Database creation & table schema (db/predictions.db)
- Audit logging of all single and batch predictions
- Historical aggregation queries:
  - Daily inspection volume
  - Category-wise adulteration rates
  - Operational risk band distributions
  - Full audit export

CRITICAL DISCLAIMER:
SYNTHETIC DATA — generated for academic demonstration.
NOT a substitute for accredited laboratory testing.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, Any

DB_DIR = "db"
DEFAULT_DB_PATH = os.path.join(DB_DIR, "predictions.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates and returns an active SQLite database connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the SQLite predictions audit table and indices if they do not exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        food_category TEXT NOT NULL,
        moisture_content REAL NOT NULL,
        ph_level REAL NOT NULL,
        storage_temperature REAL NOT NULL,
        expiry_days_remaining INTEGER NOT NULL,
        preservatives_used TEXT NOT NULL,
        color_score INTEGER NOT NULL,
        smell_score INTEGER NOT NULL,
        taste_score INTEGER NOT NULL,
        packaging_quality_score INTEGER NOT NULL,
        predicted_label INTEGER NOT NULL,
        predicted_status TEXT NOT NULL,
        probability REAL NOT NULL,
        risk_band TEXT NOT NULL
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON predictions(food_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_band ON predictions(risk_band)")

    conn.commit()
    conn.close()
    print(f"[DATABASE] Initialized SQLite audit storage at: {db_path}")


def log_prediction(sample_dict: Dict[str, Any], result_dict: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Inserts a single prediction record into the SQLite database.

    Parameters
    ----------
    sample_dict : Dict[str, Any]
        Dictionary of the 10 input parameters.
    result_dict : Dict[str, Any]
        Inference results (label, status, probability, risk_band).
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO predictions (
        timestamp, food_category, moisture_content, ph_level,
        storage_temperature, expiry_days_remaining, preservatives_used,
        color_score, smell_score, taste_score, packaging_quality_score,
        predicted_label, predicted_status, probability, risk_band
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        str(sample_dict.get('food_category')),
        float(sample_dict.get('moisture_content')),
        float(sample_dict.get('ph_level')),
        float(sample_dict.get('storage_temperature')),
        int(sample_dict.get('expiry_days_remaining')),
        str(sample_dict.get('preservatives_used')),
        int(sample_dict.get('color_score')),
        int(sample_dict.get('smell_score')),
        int(sample_dict.get('taste_score')),
        int(sample_dict.get('packaging_quality_score')),
        int(result_dict.get('predicted_label')),
        str(result_dict.get('predicted_status')),
        float(result_dict.get('probability')),
        str(result_dict.get('risk_band'))
    ))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def log_predictions_batch(df_results: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Bulk inserts a DataFrame of scored predictions into the SQLite database.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    df_to_save = df_results.copy()
    if 'timestamp' not in df_to_save.columns:
        df_to_save['timestamp'] = datetime.now().isoformat()

    cols_order = [
        'timestamp', 'food_category', 'moisture_content', 'ph_level',
        'storage_temperature', 'expiry_days_remaining', 'preservatives_used',
        'color_score', 'smell_score', 'taste_score', 'packaging_quality_score',
        'predicted_label', 'predicted_status', 'probability', 'risk_band'
    ]
    df_to_save[cols_order].to_sql('predictions', conn, if_exists='append', index=False)
    count = len(df_to_save)
    conn.close()
    return count


def get_recent_predictions(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves the most recent N logged predictions for display."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = "SELECT * FROM predictions ORDER BY id DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def get_daily_summary(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Returns inspection count, safe count, and adulterated count aggregated by date."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = """
    SELECT 
        substr(timestamp, 1, 10) as inspection_date,
        COUNT(*) as total_samples,
        SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as adulterated_count,
        SUM(CASE WHEN predicted_label = 0 THEN 1 ELSE 0 END) as safe_count,
        ROUND(AVG(probability), 4) as avg_risk_probability
    FROM predictions
    GROUP BY inspection_date
    ORDER BY inspection_date DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_category_adulteration_rate(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Returns total volume and adulteration percentage stratified by food category."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = """
    SELECT 
        food_category,
        COUNT(*) as total_inspections,
        SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) as adulterated_cases,
        ROUND(SUM(CASE WHEN predicted_label = 1 THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) as adulteration_rate_pct
    FROM predictions
    GROUP BY food_category
    ORDER BY adulteration_rate_pct DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_risk_distribution(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Returns frequency count and percentage across the three operational risk bands."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = """
    SELECT 
        risk_band,
        COUNT(*) as sample_count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predictions), 2) as percentage
    FROM predictions
    GROUP BY risk_band
    ORDER BY sample_count DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_all_predictions_as_df(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Exports all predictions from the SQLite table into a pandas DataFrame."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id ASC", conn)
    conn.close()
    return df

if __name__ == "__main__":
    init_db()
