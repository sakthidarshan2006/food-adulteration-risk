"""
app/app.py
----------
Modern Food Safety Intelligence & Adulteration Risk Inspection Dashboard
Department of Information Technology · K. Ramakrishnan College of Technology (Autonomous), Trichy

Architecture Stage 7: Food Adulteration Risk Prediction Using Data Mining Techniques
Tech Stack: Streamlit · Scikit-Learn · SQLite · Matplotlib · Seaborn

CRITICAL DISCLAIMER:
Academic demonstration only. Trained on synthetic data. NOT a substitute for
accredited laboratory testing. Do not use for real food safety decisions.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import IQRClipper
from src.predict import (
    predict_single, predict_batch, load_model_pipeline,
    determine_risk_band, compute_feature_attribution_single
)
from src.database import (
    init_db, get_recent_predictions, get_daily_summary,
    get_category_adulteration_rate, get_risk_distribution,
    get_all_predictions_as_df
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodGuard AI — Adulteration Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# High-End Design System & Custom CSS Styling
# -----------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background & Gradient Mesh */
    .stApp {
        background-color: #0b0f19;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(129, 140, 248, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
        color: #f1f5f9;
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.70) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* Permanent Academic Warning Banner */
    .warning-glass-banner {
        background: rgba(69, 26, 3, 0.55);
        border: 1px solid rgba(245, 158, 11, 0.35);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #fde68a;
        font-size: 0.88rem;
        line-height: 1.45;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.08);
    }
    .pulsing-dot {
        width: 10px;
        height: 10px;
        background-color: #f59e0b;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
        animation: pulse-amber 2s infinite;
    }
    @keyframes pulse-amber {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.70);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }

    /* Status Badges */
    .status-badge-safe {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 8px 18px;
        border-radius: 9999px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }
    .status-badge-adulterated {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 8px 18px;
        border-radius: 9999px;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.25);
    }

    /* Risk Tier Pills */
    .tier-low {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .tier-medium {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .tier-high {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Custom Gauge Bar Container */
    .gauge-container {
        background: #1e293b;
        border-radius: 9999px;
        height: 14px;
        width: 100%;
        overflow: hidden;
        margin: 10px 0;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .gauge-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Diagnostic Driver List Items */
    .driver-item {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .driver-text {
        font-size: 0.88rem;
        color: #e2e8f0;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.88rem;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Engine & Metadata
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing FoodGuard AI Engine...")
def get_cached_pipeline():
    """Loads and caches the winning model pipeline."""
    return load_model_pipeline()


@st.cache_data
def get_cached_metadata():
    """Loads model training metadata."""
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


init_db()
pipeline, metadata = get_cached_pipeline()


# -----------------------------------------------------------------------------
# Top Hero Banner Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
        <div>
            <div class="hero-title">🛡️ FoodGuard AI — Food Adulteration Risk Inspector</div>
            <div class="hero-subtitle">
                Academic Machine Learning & Chemometric Risk Intelligence · K. Ramakrishnan College of Technology (Autonomous)
            </div>
        </div>
        <div style="display: flex; gap: 10px;">
            <span style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; padding: 6px 12px; border-radius: 8px; font-size: 0.80rem; font-weight: 600;">
                ⚡ Model: Gradient Boosting
            </span>
            <span style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 6px 12px; border-radius: 8px; font-size: 0.80rem; font-weight: 600;">
                🎯 Test Recall: 100.0% (0 FN)
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Permanent Academic Disclaimer Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="warning-glass-banner">
    <span class="pulsing-dot"></span>
    <div>
        <strong>ACADEMIC SURVEILLANCE PROTOTYPE:</strong> Trained exclusively on 6,000 domain-grounded synthetic records 
        benchmarked against statutory <em>Food Safety and Standards Authority of India (FSSAI)</em> thresholds. 
        <strong>NOT</strong> a certified laboratory assay. Never use for clinical or legal enforcement decisions.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 Inspection Console")
    st.markdown("**Authority:** FSSAI Regulatory Framework")
    st.markdown("**Dataset:** 6,000 Domain-Grounded Samples")
    st.markdown("**Balancing:** Leak-Free Training SMOTE")

    st.markdown("---")
    st.markdown("#### 🚦 Statutory Risk Tiers")
    st.markdown("""
    - <span class="tier-low" style="padding: 2px 8px; font-size: 0.78rem;">🟢 Low Risk (&lt; 0.35)</span><br>
      <small style="color: #94a3b8;">Conforming commercial grade.</small>
    - <span class="tier-medium" style="padding: 2px 8px; font-size: 0.78rem;">🟡 Medium Risk (0.35–0.65)</span><br>
      <small style="color: #94a3b8;">Borderline anomalies; retest advised.</small>
    - <span class="tier-high" style="padding: 2px 8px; font-size: 0.78rem;">🔴 High Risk (&gt; 0.65)</span><br>
      <small style="color: #94a3b8;">Impound consignment & lab referral.</small>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 👥 Investigation Team")
    st.caption("Jayanth S · `2403811720521023`")
    st.caption("Jithin AK · `2403811720521025`")
    st.caption("Keerthivasan RP · `2403811720521026`")
    st.caption("Guide: B.Tech IT Faculty Board")


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Real-time Inspection",
    "📁 Batch Cargo Scanner",
    "📊 Empirical Data Insights",
    "📈 Algorithm Benchmarking",
    "🗄️ SQLite Audit Trail",
    "ℹ️ Academic Documentation"
])


# =============================================================================
# TAB 1: REAL-TIME SINGLE SAMPLE INSPECTOR
# =============================================================================
with tab1:
    st.markdown("### 🧪 Individual Consignment Inspection & Risk Profiling")
    st.markdown("Adjust measurable physicochemical parameters and organoleptic sensory scores to evaluate adulteration likelihood.")

    # Demonstration Preset Cards
    st.markdown("##### ⚡ Quick Laboratory Presets for Viva Demonstration")
    demo_c1, demo_c2, demo_c3, demo_c4 = st.columns(4)

    preset_sample = None
    with demo_c1:
        if st.button("🥛 Conforming Pure Milk", use_container_width=True, help="Fresh dairy with normal moisture and neutral pH"):
            preset_sample = {'food_category': 'milk', 'moisture_content': 87.8, 'ph_level': 6.65, 'storage_temperature': 4.0, 'expiry_days_remaining': 8, 'preservatives_used': 'none', 'color_score': 5, 'smell_score': 5, 'taste_score': 5, 'packaging_quality_score': 5}
    with demo_c2:
        if st.button("⚠️ Adulterated Milk (Water+Soda)", use_container_width=True, help="High moisture dilution + alkaline detergent shift"):
            preset_sample = {'food_category': 'milk', 'moisture_content': 93.8, 'ph_level': 7.95, 'storage_temperature': 15.0, 'expiry_days_remaining': -2, 'preservatives_used': 'non-permitted', 'color_score': 2, 'smell_score': 2, 'taste_score': 1, 'packaging_quality_score': 2}
    with demo_c3:
        if st.button("🍯 Adulterated Honey (Sugar Syrup)", use_container_width=True, help="Moisture breaches 20% FSSAI statutory threshold"):
            preset_sample = {'food_category': 'honey', 'moisture_content': 24.2, 'ph_level': 5.10, 'storage_temperature': 28.0, 'expiry_days_remaining': 90, 'preservatives_used': 'none', 'color_score': 3, 'smell_score': 3, 'taste_score': 2, 'packaging_quality_score': 4}
    with demo_c4:
        if st.button("🌶️ Dye-Adulterated Spices", use_container_width=True, help="Severe color alteration and non-permitted dye additives"):
            preset_sample = {'food_category': 'spices', 'moisture_content': 16.5, 'ph_level': 4.20, 'storage_temperature': 33.0, 'expiry_days_remaining': 12, 'preservatives_used': 'non-permitted', 'color_score': 1, 'smell_score': 2, 'taste_score': 2, 'packaging_quality_score': 2}

    # Inspection Input Form
    with st.form("modern_inspection_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            st.markdown("#### 1. Objective Physical & Chemical Measurements")

            categories = ['milk', 'oil', 'spices', 'grains', 'honey', 'beverages', 'processed']
            default_cat = preset_sample['food_category'] if preset_sample else 'milk'
            cat_choice = st.selectbox("Food Commodity Category", categories, index=categories.index(default_cat))

            m_val = float(preset_sample['moisture_content']) if preset_sample else 87.5
            moisture_in = st.slider("Moisture Content (% w/w)", 0.0, 100.0, m_val, step=0.1, help="FSSAI: Pure Milk 87-89%, Honey max 20%, Oil max 0.2%")

            ph_val = float(preset_sample['ph_level']) if preset_sample else 6.65
            ph_in = st.slider("pH Level (Hydrogen-ion Concentration)", 2.0, 10.0, ph_val, step=0.05, help="Fresh milk: 6.5-6.8. Alkaline > 7.5 indicates neutralizer/detergent.")

            temp_val = float(preset_sample['storage_temperature']) if preset_sample else 4.0
            temp_in = st.number_input("Cold-Chain / Storage Temperature (°C)", -20.0, 50.0, temp_val, step=0.5, help="Statutory cold storage limit for milk <= 6.0°C.")

            exp_val = int(preset_sample['expiry_days_remaining']) if preset_sample else 7
            exp_in = st.number_input("Shelf-Life Days Remaining (Negative = Expired)", -60, 730, exp_val, step=1, help="Expired commodities represent statutory violations.")

            pres_options = ['none', 'permitted', 'non-permitted']
            default_pres = preset_sample['preservatives_used'] if preset_sample else 'none'
            pres_in = st.selectbox("Chemical Preservative Screening Assay", pres_options, index=pres_options.index(default_pres))

        with form_col2:
            st.markdown("#### 2. Organoleptic Sensory Ratings (1 to 5 Likert)")
            st.caption("1 = Acute Chemical Defect / Severe Anomaly | 5 = Conforming Commercial Grade")

            c_def = int(preset_sample['color_score']) if preset_sample else 5
            color_in = st.select_slider("Color Score (Discoloration / Synthetic Dyes)", options=[1, 2, 3, 4, 5], value=c_def)

            s_def = int(preset_sample['smell_score']) if preset_sample else 5
            smell_in = st.select_slider("Smell Score (Chemical / Rancidity / Sourness)", options=[1, 2, 3, 4, 5], value=s_def)

            t_def = int(preset_sample['taste_score']) if preset_sample else 5
            taste_in = st.select_slider("Taste Score (Unpalatable / Acrid / Off-flavor)", options=[1, 2, 3, 4, 5], value=t_def)

            p_def = int(preset_sample['packaging_quality_score']) if preset_sample else 5
            pkg_in = st.select_slider("Packaging Integrity (Tampered Seal / Swollen)", options=[1, 2, 3, 4, 5], value=p_def)

        eval_btn = st.form_submit_button("⚡ Execute Forensic Adulteration Assessment", use_container_width=True)

    if eval_btn:
        sample_payload = {
            'food_category': cat_choice,
            'moisture_content': moisture_in,
            'ph_level': ph_in,
            'storage_temperature': temp_in,
            'expiry_days_remaining': exp_in,
            'preservatives_used': pres_in,
            'color_score': color_in,
            'smell_score': smell_in,
            'taste_score': taste_in,
            'packaging_quality_score': pkg_in
        }

        # Run inference through pipeline
        inf_result = predict_single(sample_payload, log_to_db=True)
        prob = inf_result['probability']
        status = inf_result['predicted_status']
        risk_tier = inf_result['risk_band']
        db_id = inf_result.get('database_log_id', 'Logged')

        st.markdown("---")
        st.markdown("### 📋 Diagnostic Inspection Verdict")

        # Top Diagnostic Metric Cards
        diag_c1, diag_c2, diag_c3, diag_c4 = st.columns(4)

        with diag_c1:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Safety Status</div>', unsafe_allow_html=True)
            if status == "Safe":
                st.markdown('<div style="margin-top: 8px;"><span class="status-badge-safe">🛡️ SAFE</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="margin-top: 8px;"><span class="status-badge-adulterated">🚨 ADULTERATED</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with diag_c2:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Risk Probability</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{prob:.1%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with diag_c3:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Operational Risk Band</div>', unsafe_allow_html=True)
            if risk_tier == "Low Risk":
                st.markdown('<div style="margin-top: 8px;"><span class="tier-low">🟢 LOW RISK</span></div>', unsafe_allow_html=True)
            elif risk_tier == "Medium Risk":
                st.markdown('<div style="margin-top: 8px;"><span class="tier-medium">🟡 MEDIUM RISK</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="margin-top: 8px;"><span class="tier-high">🔴 HIGH RISK</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with diag_c4:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">SQLite Audit Provenance</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value" style="font-size: 1.5rem; color: #38bdf8;">#{db_id} <span style="font-size: 0.8rem; color: #94a3b8;">(Verified)</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # High-Tech Gradient Risk Gauge Bar
        st.markdown("##### 🧭 Calibrated Risk Probability Gauge")
        gauge_color = "#10b981" if prob < 0.35 else ("#f59e0b" if prob <= 0.65 else "#ef4444")
        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-fill" style="width: {min(100.0, max(2.0, prob * 100))}%; background: linear-gradient(90deg, #10b981 0%, {gauge_color} 100%);"></div>
        </div>
        """, unsafe_allow_html=True)

        gc1, gc2, gc3 = st.columns([0.35, 0.30, 0.35])
        gc1.markdown("<small style='color: #34d399;'>🟢 <strong>Low Risk:</strong> 0.00 – 0.35 (Clear Consignment)</small>", unsafe_allow_html=True)
        gc2.markdown("<small style='color: #fbbf24;'>🟡 <strong>Medium Risk:</strong> 0.35 – 0.65 (Retest Advised)</small>", unsafe_allow_html=True)
        gc3.markdown("<small style='color: #f87171;'>🔴 <strong>High Risk:</strong> 0.65 – 1.00 (Immediate Lab Referral)</small>", unsafe_allow_html=True)

        # Actionable Operational Guidance Box
        st.markdown("---")
        if risk_tier == "Low Risk":
            st.success("✅ **STATUTORY DISPOSITION:** All physical, chemical, and organoleptic indicators conform to FSSAI baseline standards. Consignment cleared for open commercial distribution.")
        elif risk_tier == "Medium Risk":
            st.warning("⚠️ **STATUTORY DISPOSITION: RETEST ADVISED.** Sample displays borderline organoleptic or thermal anomalies. Hold consignment for secondary field screening or rapid enzymatic assay prior to market clearance.")
        else:
            st.error("🚨 **STATUTORY DISPOSITION: HIGH RISK — IMMEDIATE SEIZURE RECOMMENDED.** Severe chemical or physical threshold violation detected. Impound lot immediately and dispatch duplicate sealed forensic samples to an accredited referral laboratory under Section 47 of the FSS Act, 2006.")

        # Explainable AI (XAI) Diagnostic Drivers
        st.markdown("#### 🧠 Explainable AI: Detected Regulatory Non-Conformance Factors")
        drivers = inf_result.get('risk_drivers', [])
        for d in drivers:
            icon = "🔴" if d['severity'] == "High" else ("🟡" if d['severity'] == "Medium" else "🟢")
            st.markdown(f"""
            <div class="driver-item">
                <span class="driver-text">{icon} <strong>{d['feature']}:</strong> {d['detail']}</span>
                <span style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: {'#f87171' if d['severity'] == 'High' else ('#fbbf24' if d['severity'] == 'Medium' else '#34d399')};">{d['status']}</span>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: BATCH CARGO SCANNER
# =============================================================================
with tab2:
    st.markdown("### 🚢 High-Throughput Cargo Screening & Batch Auditing")
    st.markdown("Upload bulk shipment manifests or regional market surveillance CSV files to execute high-speed automated triage.")

    # Template Download
    demo_df = pd.DataFrame([
        {'food_category': 'milk', 'moisture_content': 87.8, 'ph_level': 6.65, 'storage_temperature': 4.0, 'expiry_days_remaining': 10, 'preservatives_used': 'none', 'color_score': 5, 'smell_score': 5, 'taste_score': 5, 'packaging_quality_score': 5},
        {'food_category': 'milk', 'moisture_content': 93.2, 'ph_level': 7.85, 'storage_temperature': 14.0, 'expiry_days_remaining': -2, 'preservatives_used': 'non-permitted', 'color_score': 2, 'smell_score': 2, 'taste_score': 1, 'packaging_quality_score': 2},
        {'food_category': 'oil', 'moisture_content': 0.12, 'ph_level': 6.10, 'storage_temperature': 22.0, 'expiry_days_remaining': 180, 'preservatives_used': 'none', 'color_score': 5, 'smell_score': 5, 'taste_score': 5, 'packaging_quality_score': 5},
        {'food_category': 'honey', 'moisture_content': 23.5, 'ph_level': 5.20, 'storage_temperature': 26.0, 'expiry_days_remaining': 60, 'preservatives_used': 'none', 'color_score': 3, 'smell_score': 3, 'taste_score': 2, 'packaging_quality_score': 4},
        {'food_category': 'spices', 'moisture_content': 17.0, 'ph_level': 4.10, 'storage_temperature': 34.0, 'expiry_days_remaining': 10, 'preservatives_used': 'non-permitted', 'color_score': 1, 'smell_score': 2, 'taste_score': 2, 'packaging_quality_score': 2}
    ])

    st.download_button(
        "📥 Download Verified Surveillance CSV Template",
        data=demo_df.to_csv(index=False),
        file_name="food_inspection_manifest_template.csv",
        mime="text/csv",
        help="Use this template containing all 10 required parameter headers"
    )

    batch_file = st.file_uploader("Upload Shipment Manifest CSV", type=['csv'])

    if batch_file is not None:
        try:
            input_df = pd.read_csv(batch_file)
            st.info(f"Loaded **{len(input_df)}** cargo samples. Executing batch vectorized classification...")

            with st.spinner("Processing samples through pipeline and updating SQLite audit log..."):
                scored_batch = predict_batch(input_df, log_to_db=True)

            # High-Level Metrics
            bk1, bk2, bk3, bk4 = st.columns(4)
            bk1.metric("Total Samples Scanned", len(scored_batch))
            bk2.metric("Safe Cleared", int((scored_batch['predicted_label'] == 0).sum()))
            bk3.metric("Adulterated Flagged", int((scored_batch['predicted_label'] == 1).sum()))
            bk4.metric("Adulteration Prevalence", f"{(scored_batch['predicted_label'] == 1).mean():.1%}")

            st.markdown("##### Scored Shipment Inspection Log")
            st.dataframe(
                scored_batch[['food_category', 'moisture_content', 'ph_level', 'storage_temperature', 'predicted_status', 'probability', 'risk_band']],
                use_container_width=True
            )

            csv_out = scored_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Scored Cargo Results (CSV)",
                data=csv_out,
                file_name="scored_cargo_inspection_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as err:
            st.error(f"Batch Processing Error: {err}")


# =============================================================================
# TAB 3: EMPIRICAL DATA INSIGHTS & EDA
# =============================================================================
with tab3:
    st.markdown("### 📊 Exploratory Data Analysis & Empirical Distributions")
    st.markdown("Publication-grade statistical visualizations demonstrating parameter distributions, commodity baselines, and adulterant physics.")

    fig_catalog = {
        "Figure 1: Target Class Balance (Safe vs Adulterated)": "fig1_class_balance.png",
        "Figure 2: Probability Density of pH by Adulteration Class": "fig2_ph_distribution_by_class.png",
        "Figure 3: Moisture Content across Food Matrix": "fig3_moisture_by_food_category.png",
        "Figure 4: Inter-Parameter Correlation Matrix Heatmap": "fig4_correlation_heatmap.png",
        "Figure 5: Sensory & Packaging Degradation Boxplots": "fig5_sensory_score_boxplots.png",
        "Figure 6: Adulteration Prevalence Rate across Categories": "fig6_adulteration_rate_by_category.png",
        "Figure 7: Storage Temperature vs Continuous Risk": "fig7_storage_temperature_vs_risk.png",
        "Figure 8: Expiry Days Remaining by Adulteration Status": "fig8_expiry_vs_risk.png"
    }

    selected_title = st.selectbox("Select Visual Assay to Examine", list(fig_catalog.keys()))
    fname = fig_catalog[selected_title]
    fpath = os.path.join(PROJECT_ROOT, "reports", "figures", fname)

    if os.path.exists(fpath):
        st.image(fpath, use_container_width=True)

        interpretations_dict = {
            "fig1_class_balance.png": "Figure 1 illustrates the empirical class distribution of the primary target is_adulterated across the 6,000 synthesized samples. Conforming safe food samples constitute 3,983 records (66.38%), whereas adulterated/contaminated samples represent 2,017 records (33.62%). This 2:1 majority-to-minority ratio closely models real-world food safety surveillance scenarios where adulterated commodities remain a minority. This intentional class imbalance serves as the direct mathematical motivation for applying SMOTE oversampling exclusively on the training partition in Stage 2 to prevent algorithmic bias without incurring synthetic test leakage.",
            "fig2_ph_distribution_by_class.png": "Figure 2 displays Kernel Density Estimation (KDE) curves of hydrogen ion concentration (pH) stratified by adulteration status. Safe samples exhibit sharp, well-defined multimodal peaks centered around commodity-specific physiological equilibriums—specifically near pH 3.3 for acidic fruit juices/honey, pH 5.8 for spices/processed foods, and pH 6.65 for fresh dairy. In contrast, adulterated samples exhibit high dispersion with pronounced tails stretching into severe alkaline regions (pH > 7.5, indicative of detergent or caustic soda addition in milk) and abnormal acidic regions (pH < 4.8 in milk, indicative of bacterial souring and curdling due to cold chain breakdown).",
            "fig3_moisture_by_food_category.png": "Figure 3 demonstrates moisture distribution across all seven commodity matrices split by class status. In commodities with strictly bounded physiological water content—most notably milk (FSSAI benchmark ~87.5%) and honey (FSSAI statutory maximum 20.0%)—adulterated samples exhibit statistically significant upward shifts. In milk, dilution with extraneous water pushes moisture toward 91–96%, while in honey, inverted sugar syrups breach the 20% limit into the 22–28% range. Conversely, edible oils maintain near-zero baseline moisture (<0.2%), where any sample breaching 0.5% moisture strongly signals cheap adulterant blending or improper dehydration.",
            "fig4_correlation_heatmap.png": "The Pearson correlation heatmap in Figure 4 reveals strong, intuitive collinearities between sensory degradation indicators and regulatory risk. Specifically, taste score (r = -0.94), smell score (r = -0.90), color score (r = -0.90), and packaging quality (r = -0.89) exhibit strong negative correlations with adulteration status, validating that organoleptic failure is the primary frontline symptom of contamination. Shelf-life remaining (expiry_days_remaining) exhibits a robust negative correlation (r = -0.68), confirming that expired batches correlate heavily with spoilage.",
            "fig5_sensory_score_boxplots.png": "Figure 5 breaks down the four ordinal 1–5 inspection scores across safe and adulterated cohorts. Safe products consistently cluster at median scores of 5 (prime) with a 25th percentile at 4 (acceptable), demonstrating tight conformity. Adulterated commodities, however, show interquartile spans dropping sharply into the 1–3 range, with medians centered at 2 (moderate defect). Packaging integrity shows lower variance in safe products compared to high variance in adulterated stock, where counterfeit labeling, tampered seals, and swollen cans are frequent physical flags.",
            "fig6_adulteration_rate_by_category.png": "Figure 6 depicts the proportion of non-conforming samples identified within each specific food commodity sector. While baseline category distribution was weighted across sampling priority, adulteration prevalence remains uniformly realistic across all classes, ranging from ~31% to ~36%. Spices and dairy exhibit marginally higher non-conformance rates (35.4% and 34.8% respectively), mirroring historical FSSAI surveillance data where high economic value and liquid/powdered states make milk (water/starch/detergent) and powdered spices (synthetic dyes/sawdust) prime targets for commercial profit-motivated adulteration.",
            "fig7_storage_temperature_vs_risk.png": "Figure 7 plots storage temperature against continuous risk scores, color-coded by the three operational risk bands. For perishable chilled commodities (such as pasteurized milk and processed juices), temperatures exceeding statutory cold-chain limits (>8°C) cause a steep parabolic escalation into High Risk (>0.65). Ambient commodities (grains, spices, oils) tolerate 20–25°C, but samples stored under extreme tropical heat abuse (>35°C) exhibit accelerated lipid rancidity, pest proliferation, and chemical preservative breakdown, pushing risk scores across the 0.35 medium threshold.",
            "fig8_expiry_vs_risk.png": "Figure 8 visualizes the histogram distribution of shelf-life days remaining, highlighting the critical inflection boundary at Day 0 (statutory expiration). Samples to the left of the red dashed line (negative days remaining) are virtually 100% adulterated/non-compliant under Section 23 of the FSS Act, as selling expired stock constitutes illegal contamination regardless of organoleptic condition. Furthermore, a dense cluster of contaminated samples appears in the 0 to 10 day window, capturing near-expiry inventory dumping by vendors."
        }

        st.info(f"**Analytical Interpretation:**\n\n{interpretations_dict.get(fname, 'Detailed analytical interpretation documented in Notebook 03.')}")


# =============================================================================
# TAB 4: ALGORITHM BENCHMARKING
# =============================================================================
with tab4:
    st.markdown("### 📈 Multi-Model Benchmarking & Safety-First Evaluation")
    st.markdown("Empirical comparison of all 7 machine learning models across 5-fold cross-validation and hold-out test metrics.")

    # Comparison Table
    b_table = [
        {"Classifier": "Gradient Boosting (Winning Model)", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "1.0000", "5-Fold CV Recall": "0.9997 ± 0.0006"},
        {"Classifier": "Random Forest", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9999", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Classifier": "Decision Tree", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9998", "5-Fold CV Recall": "0.9975 ± 0.0032"},
        {"Classifier": "Support Vector Machine (RBF)", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9985", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Classifier": "K-Nearest Neighbors (k=7)", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9980", "5-Fold CV Recall": "0.9997 ± 0.0006"},
        {"Classifier": "Logistic Regression", "Accuracy": "99.75%", "Precision": "99.50%", "Recall (Adulterated)": "99.75%", "F1-Score": "0.9963", "ROC-AUC": "0.9999", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Classifier": "Gaussian Naive Bayes", "Accuracy": "99.67%", "Precision": "99.26%", "Recall (Adulterated)": "99.75%", "F1-Score": "0.9950", "ROC-AUC": "0.9993", "5-Fold CV Recall": "1.0000 ± 0.0000"}
    ]
    st.dataframe(pd.DataFrame(b_table), use_container_width=True)

    st.markdown("""
    > 🛡️ **Academic Justification: Why Model Selection Prioritizes RECALL Over Accuracy:**  
    > In food safety, a **False Negative** means releasing toxic or adulterated food into supermarkets, causing toxic poisoning or fatalities.  
    > In contrast, a **False Positive** merely refers a safe consignment for secondary lab assay. Therefore, our winning pipeline was selected based on **Recall = 100.0% (Zero False Negatives)** on hold-out testing.
    """)

    # Visual Diagnostics Columns
    diag_c1, diag_c2 = st.columns(2)
    with diag_c1:
        st.markdown("##### Overlaid ROC Curves")
        roc_f = os.path.join(PROJECT_ROOT, "reports", "figures", "fig10_model_comparison_roc.png")
        if os.path.exists(roc_f):
            st.image(roc_f, use_container_width=True)

    with diag_c2:
        st.markdown("##### Confusion Matrix Grid")
        cm_f = os.path.join(PROJECT_ROOT, "reports", "figures", "fig11_confusion_matrices.png")
        if os.path.exists(cm_f):
            st.image(cm_f, use_container_width=True)

    st.markdown("##### Global Feature Importance Ranking (Tree Ensemble)")
    feat_f = os.path.join(PROJECT_ROOT, "reports", "figures", "fig12_shap_feature_importance.png")
    if os.path.exists(feat_f):
        st.image(feat_f, use_container_width=True)

    st.markdown("---")
    st.subheader("Apriori Association Rule Mining: Top 10 High-Lift Decision Patterns")
    apriori_data = [
        {"Antecedents": "NonPermitted_Preservative", "Consequents": "Adulterated", "Support": "15.92%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "NonPermitted_Preservative & Expired_Stock", "Consequents": "Adulterated", "Support": "9.32%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "NonPermitted_Preservative & Severe_Sensory_Defect", "Consequents": "Adulterated", "Support": "15.30%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "NonPermitted_Preservative & Packaging_Compromised", "Consequents": "Adulterated", "Support": "10.97%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "NonPermitted_Preservative & Storage_Temp_Abuse", "Consequents": "Adulterated", "Support": "6.73%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "NonPermitted_Preservative & Moisture_Anomalous", "Consequents": "Adulterated", "Support": "8.20%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "Expired_Stock & Moisture_Anomalous", "Consequents": "Adulterated", "Support": "10.70%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "Expired_Stock & pH_Anomalous", "Consequents": "Adulterated", "Support": "6.00%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "Packaging_Compromised & pH_Anomalous", "Consequents": "Adulterated", "Support": "6.52%", "Confidence": "100.0%", "Lift": "2.97"},
        {"Antecedents": "Severe_Sensory_Defect & pH_Anomalous", "Consequents": "Adulterated", "Support": "11.85%", "Confidence": "99.86%", "Lift": "2.97"}
    ]
    st.table(pd.DataFrame(apriori_data))


# =============================================================================
# TAB 5: SQLITE AUDIT TRAIL
# =============================================================================
with tab5:
    st.markdown("### 🗄️ Regulatory Audit Trail & Longitudinal Surveillance")
    st.markdown("All inspections performed via the dashboard are permanently logged into SQLite database `db/predictions.db` for statutory chain of custody.")

    df_hist = get_all_predictions_as_df()

    if len(df_hist) > 0:
        hk1, hk2, hk3, hk4 = st.columns(4)
        hk1.metric("Total Inspections Logged", len(df_hist))
        hk2.metric("Adulterated Impounded", int((df_hist['predicted_label'] == 1).sum()))
        hk3.metric("Safe Cleared", int((df_hist['predicted_label'] == 0).sum()))
        hk4.metric("Mean Predicted Risk", f"{df_hist['probability'].mean():.1%}")

        st.markdown("---")
        h_col1, h_col2 = st.columns(2)

        with h_col1:
            st.markdown("##### Longitudinal Adulteration by Food Category")
            cat_stats = df_hist.groupby('food_category')['predicted_label'].mean() * 100
            st.bar_chart(cat_stats)

        with h_col2:
            st.markdown("##### Distribution of Assigned Risk Tiers")
            tier_counts = df_hist['risk_band'].value_counts()
            st.bar_chart(tier_counts)

        st.markdown("##### Recent Inspection Records")
        recent_records = get_recent_predictions(limit=50)
        st.dataframe(recent_records, use_container_width=True)

        full_csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Export Entire SQLite Inspection Database (CSV)",
            data=full_csv,
            file_name=f"sqlite_audit_export_{len(df_hist)}_records.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No inspection entries recorded yet. Test an inspection in Tab 1 or batch upload in Tab 2 to populate the database.")


# =============================================================================
# TAB 6: ACADEMIC DOCUMENTATION
# =============================================================================
with tab6:
    st.markdown("### 🏛️ Academic Institutional Details & Methodology")

    st.markdown("""
    #### 🎓 Course & Affiliation
    - **Course Title:** Data Mining Techniques (DMT)
    - **Degree:** Bachelor of Technology in Information Technology
    - **Institution:** K. Ramakrishnan College of Technology (Autonomous), Trichy, Tamil Nadu, India
    - **Academic Year:** 2024 – 2025

    #### 👥 Project Authors
    - **Jayanth S** (Univ. Reg. No: `2403811720521023`)
    - **Jithin AK** (Univ. Reg. No: `2403811720521025`)
    - **Keerthivasan RP** (Univ. Reg. No: `2403811720521026`)

    ---

    #### 📚 Published Statutory Standards Referenced
    1. **Food Safety and Standards Act, 2006** (Act No. 34 of 2006), Government of India.
    2. **FSSAI Food Safety and Standards (Food Products Standards) Regulations, 2011.**
    3. **FSSAI Manual of Methods of Analysis of Foods (Milk and Milk Products), 2016.**
    4. **FSSAI National Milk Safety Survey (NMSS), 2019.**
    5. **Bureau of Indian Standards (BIS) IS 6273:** Sensory Evaluation of Food.

    ---

    #### 🔍 Known Methodological Weaknesses & Future Scope
    - **Human Sensory Limitation:** Sensory ratings (color, smell, taste, packaging) are subjective 1–5 scales carrying inter-rater variability and olfactory fatigue.
    - **IoT & Spectroscopy Enhancement:** Future iterations will interface handheld Near-Infrared (NIR) spectrometers (900–1700 nm) and Metal-Oxide Semiconductor (MOS) Electronic-Nose gas sensor arrays to replace human sensory scores with objective digitized chemometric feature vectors.
    """)
