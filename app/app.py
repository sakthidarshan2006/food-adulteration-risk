"""
app/app.py
----------
FoodGuard AI — Forensic Food Adulteration Risk Intelligence Cockpit
Department of Information Technology · K. Ramakrishnan College of Technology (Autonomous), Trichy

Unique Architecture:
- Dynamic Multi-Chamber Forensic Lab Bench (Replaces standard form layouts)
- Commodity-Adaptive Statutory Threshold HUD (Dynamically adapts limits per food matrix)
- Dual-Dial Radial SVG Risk Speedometer & Statutory Action Directive
- Workflow Control Hub (Replaces generic tabs with specialized operational consoles)
- Live SQLite Audit Ledger & Longitudinal Surveillance Analytics

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
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodGuard AI — Forensic Cockpit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Custom Advanced UI Design System (Forensic Lab HUD)
# -----------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #080c14;
        background-image: 
            radial-gradient(circle at 15% 10%, rgba(14, 165, 233, 0.09) 0px, transparent 40%),
            radial-gradient(circle at 85% 15%, rgba(99, 102, 241, 0.08) 0px, transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(16, 185, 129, 0.06) 0px, transparent 50%);
        color: #e2e8f0;
    }

    /* Top Command Header */
    .command-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.90) 0%, rgba(30, 41, 59, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px 28px;
        margin-bottom: 18px;
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 14px;
    }
    .hud-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
    }
    .hud-sub {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 500;
        margin-top: 3px;
    }

    /* Academic Warning Beacon */
    .statutory-banner {
        background: rgba(69, 26, 3, 0.6);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #fef08a;
        font-size: 0.86rem;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.07);
    }
    .pulse-dot {
        width: 9px;
        height: 9px;
        background-color: #f59e0b;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
        animation: pulse-ring 2s infinite;
    }
    @keyframes pulse-ring {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }

    /* Chamber Panels (Unique 3-Chamber Inspection Layout) */
    .chamber-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .chamber-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .chamber-cyan::before { background: linear-gradient(90deg, #0ea5e9, #38bdf8); }
    .chamber-purple::before { background: linear-gradient(90deg, #6366f1, #a855f7); }
    .chamber-emerald::before { background: linear-gradient(90deg, #10b981, #34d399); }

    .chamber-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .chamber-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .chamber-tag {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.08);
        color: #94a3b8;
    }

    /* Commodity Threshold Guidance Pill */
    .norm-pill {
        background: rgba(56, 189, 248, 0.08);
        border: 1px dashed rgba(56, 189, 248, 0.3);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #bae6fd;
        margin-bottom: 14px;
        line-height: 1.45;
    }

    /* Litmus Scale Bar */
    .litmus-bar {
        height: 6px;
        border-radius: 4px;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 25%, #10b981 50%, #06b6d4 75%, #8b5cf6 100%);
        margin: 6px 0 12px 0;
        opacity: 0.85;
    }

    /* Result Triage HUD */
    .verdict-cockpit {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 24px 28px;
        margin-top: 20px;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4);
    }

    /* Status Glow Orbs */
    .glow-safe {
        color: #34d399;
        font-weight: 800;
        font-size: 1.65rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    .glow-adulterated {
        color: #f87171;
        font-weight: 800;
        font-size: 1.65rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }

    /* Circular SVG Speedometer */
    .speedometer-svg {
        display: block;
        margin: 0 auto;
    }

    /* Forensic Driver Pill */
    .forensic-flag {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
    }
    .forensic-flag-warning {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
    }
    .forensic-flag-conforming {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Pipeline & Metadata
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Engaging Forensic Machine Learning Engine...")
def get_cached_pipeline():
    return load_model_pipeline()


@st.cache_data
def get_cached_metadata():
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


init_db()
pipeline, metadata = get_cached_pipeline()


# -----------------------------------------------------------------------------
# Top Command Header & Institutional Branding
# -----------------------------------------------------------------------------
st.markdown("""
<div class="command-header">
    <div>
        <div class="hud-title">🛡️ FoodGuard AI — Food Adulteration Risk Inspector</div>
        <div class="hud-sub">
            Forensic Decision-Support System · Dept of Information Technology · K. Ramakrishnan College of Technology (Autonomous), Trichy
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.4); color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; padding: 5px 12px; border-radius: 6px; font-weight: 600;">
            ENGINE: Gradient Boosting
        </span>
        <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; padding: 5px 12px; border-radius: 6px; font-weight: 600;">
            TEST RECALL: 100.0% (0 FN)
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Permanent Statutory Academic Warning Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="statutory-banner">
    <span class="pulse-dot"></span>
    <div>
        <strong>STATUTORY NOTICE (ACADEMIC BENCHMARK):</strong> Engineered strictly for educational evaluation in Data Mining Techniques. 
        Trained on 6,000 domain-synthesized samples grounded in <em>Food Safety and Standards Authority of India (FSSAI)</em> specifications. 
        <strong>NOT</strong> an accredited chemical laboratory assay. Do not execute real legal seizures or medical dietary clearances based solely on this tool.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Modern Top Navigation Deck (Unique Workflow Selector)
# -----------------------------------------------------------------------------
menu_choice = st.radio(
    "Select Operational Inspection Mode",
    [
        "🔬 Forensic Inspection Deck",
        "🚢 High-Volume Cargo Audit",
        "📊 Empirical Surveillance Insights",
        "📈 Algorithmic Benchmark Lab",
        "🗄️ SQLite Chain-of-Custody Ledger",
        "ℹ️ Institutional Charter & Standards"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# =============================================================================
# MODE 1: FORENSIC INSPECTION DECK (RADICALLY UNIQUE 3-CHAMBER HUD)
# =============================================================================
if menu_choice == "🔬 Forensic Inspection Deck":
    st.markdown("### 🔬 Multi-Chamber Forensic Assessment Deck")
    st.markdown("Configure the sample profile across three specialized analytical chambers to compute real-time statutory risk.")

    # 1. Commodity Selector Bar
    c_bar1, c_bar2 = st.columns([0.40, 0.60])

    with c_bar1:
        commodities = ['milk', 'oil', 'spices', 'grains', 'honey', 'beverages', 'processed']
        commodity_icons = {
            'milk': '🥛 Milk & Liquid Dairy',
            'oil': '🛢️ Edible Oils & Lipids',
            'spices': '🌶️ Condiments & Spices',
            'grains': '🌾 Cereals & Grains',
            'honey': '🍯 Raw & Processed Honey',
            'beverages': '🥤 Juices & Soft Drinks',
            'processed': '📦 Bakery & Ready-to-Eat'
        }
        chosen_cat = st.selectbox(
            "Selected Food Commodity Matrix",
            commodities,
            format_func=lambda x: commodity_icons.get(x, x.title())
        )

    # Commodity Statutory Tolerance Metadata Card
    commodity_norms = {
        'milk': "📌 <strong>FSSAI Milk Standards:</strong> Normal Moisture 87.0% – 89.0% | pH 6.50 – 6.80 | Max Cold Chain Temp &le; 6.0°C | Alkaline shift (>7.5) signals caustic detergent.",
        'oil': "📌 <strong>FSSAI Edible Oils:</strong> Moisture &le; 0.20% | Pure pH 5.50 – 6.50 | Shelf Temp 18°C – 28°C | Moisture >0.25% indicates water/mineral oil dilution.",
        'spices': "📌 <strong>FSSAI Spices Regulations:</strong> Moisture 6.0% – 12.0% | pH 5.00 – 6.40 | Zero tolerance for non-permitted coal-tar dyes (Metanil Yellow, Sudan dyes).",
        'grains': "📌 <strong>FSSAI Grain Tolerances:</strong> Safe Moisture 10.0% – 14.0% | pH 6.00 – 6.80 | Excess moisture (>14.5%) sparks hazardous Aspergillus fungal aflatoxins.",
        'honey': "📌 <strong>FSSAI Honey Standards (2018):</strong> Statutory Maximum Moisture strictly &le; 20.0% | Acidic pH 3.50 – 4.50 | Moisture >20% proves inverted sugar syrup adulteration.",
        'beverages': "📌 <strong>FSSAI Beverage Norms:</strong> Moisture 85.0% – 94.0% | Acidic pH 2.80 – 4.20 | Storage &le; 12.0°C | Chemical off-flavor signals synthetic sweetener fraud.",
        'processed': "📌 <strong>FSSAI Processed Foods:</strong> Moisture 6.0% – 18.0% | pH 4.50 – 6.50 | Sensory score degradation signals recycled rancid frying fats."
    }

    with c_bar2:
        st.markdown(f"<div class='norm-pill'>{commodity_norms.get(chosen_cat, '')}</div>", unsafe_allow_html=True)

    # Quick Lab Pre-sets for Viva Voce Testing
    st.markdown("##### ⚡ Quick Laboratory Presets for Viva Demonstration")
    q_c1, q_c2, q_c3, q_c4 = st.columns(4)
    active_preset = None

    with q_c1:
        if st.button("🥛 Conforming Pure Milk", use_container_width=True):
            chosen_cat = 'milk'
            active_preset = {'moisture': 87.8, 'ph': 6.65, 'temp': 4.0, 'exp': 8, 'pres': 'none', 'c': 5, 's': 5, 't': 5, 'p': 5}
    with q_c2:
        if st.button("⚠️ Adulterated Milk (Water+Soda)", use_container_width=True):
            chosen_cat = 'milk'
            active_preset = {'moisture': 93.6, 'ph': 7.95, 'temp': 15.0, 'exp': -1, 'pres': 'non-permitted', 'c': 2, 's': 2, 't': 1, 'p': 2}
    with q_c3:
        if st.button("🍯 Adulterated Honey (High Sugar)", use_container_width=True):
            chosen_cat = 'honey'
            active_preset = {'moisture': 24.5, 'ph': 5.20, 'temp': 28.0, 'exp': 90, 'pres': 'none', 'c': 3, 's': 3, 't': 2, 'p': 4}
    with q_c4:
        if st.button("🌶️ Dye-Adulterated Spices", use_container_width=True):
            chosen_cat = 'spices'
            active_preset = {'moisture': 16.5, 'ph': 4.15, 'temp': 33.0, 'exp': 14, 'pres': 'non-permitted', 'c': 1, 's': 2, 't': 2, 'p': 2}

    # -------------------------------------------------------------------------
    # Unique 3-Chamber Inspection Layout (Radically Different Input Partition)
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns([1, 1.1, 1.1])

    with ch1:
        st.markdown("""
        <div class="chamber-card chamber-cyan">
            <div class="chamber-header">
                <span style="font-size: 1.3rem;">❄️</span>
                <div>
                    <div class="chamber-title">Chamber 1: Cold Chain & Chronometry</div>
                    <div class="chamber-tag">Telemetry & Logistics</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        temp_default = active_preset['temp'] if active_preset else (4.0 if chosen_cat == 'milk' else 22.0)
        temp_input = st.number_input("Cold Storage Temperature (°C)", -20.0, 50.0, float(temp_default), step=0.5)

        # Thermal condition indicator
        if temp_input <= 6.0:
            st.caption("🟢 **Thermal Status:** Optimal Chilled Chain")
        elif temp_input <= 25.0:
            st.caption("🟡 **Thermal Status:** Ambient Room Storage")
        else:
            st.caption("🔴 **Thermal Status:** Severe Thermal Heat Abuse")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        exp_default = active_preset['exp'] if active_preset else 15
        exp_input = st.number_input("Shelf-Life Remaining (Days)", -60, 730, int(exp_default), step=1)

        if exp_input <= 0:
            st.error(f"⚠️ Expired by {abs(exp_input)} days (Statutory Prohibition)")
        elif exp_input <= 5:
            st.warning("⚠️ Critical: Approaching Expiry (< 5 days)")
        else:
            st.success("✅ Commercial Shelf-Life Valid")

        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        st.markdown("""
        <div class="chamber-card chamber-purple">
            <div class="chamber-header">
                <span style="font-size: 1.3rem;">🧪</span>
                <div>
                    <div class="chamber-title">Chamber 2: Physicochemical Sensor Lab</div>
                    <div class="chamber-tag">Chemical Assays</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        ph_default = active_preset['ph'] if active_preset else (6.65 if chosen_cat == 'milk' else (3.9 if chosen_cat == 'honey' else 6.0))
        ph_input = st.slider("Digital pH Electrode (Acidity / Alkalinity)", 2.0, 10.0, float(ph_default), step=0.05)
        st.markdown("<div class='litmus-bar'></div>", unsafe_allow_html=True)

        moisture_default = active_preset['moisture'] if active_preset else (87.5 if chosen_cat == 'milk' else (18.0 if chosen_cat == 'honey' else (0.1 if chosen_cat == 'oil' else 11.0)))
        moisture_input = st.slider("Gravimetric Moisture Content (% w/w)", 0.0, 100.0, float(moisture_default), step=0.1)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        pres_opts = ['none', 'permitted', 'non-permitted']
        pres_default = active_preset['pres'] if active_preset else 'none'
        pres_input = st.selectbox("Chemical Preservative Assay", pres_opts, index=pres_opts.index(pres_default))

        st.markdown("</div>", unsafe_allow_html=True)

    with ch3:
        st.markdown("""
        <div class="chamber-card chamber-emerald">
            <div class="chamber-header">
                <span style="font-size: 1.3rem;">👁️</span>
                <div>
                    <div class="chamber-title">Chamber 3: Organoleptic Sensory Cockpit</div>
                    <div class="chamber-tag">Likert Matrix 1–5</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            c_val = active_preset['c'] if active_preset else 5
            color_in = st.select_slider("Color Score", options=[1, 2, 3, 4, 5], value=int(c_val), help="1 = Severe Dye Discoloration | 5 = Natural")

            s_val = active_preset['s'] if active_preset else 5
            smell_in = st.select_slider("Aroma Score", options=[1, 2, 3, 4, 5], value=int(s_val), help="1 = Foul / Rancid | 5 = Fresh Aroma")

        with col_c2:
            t_val = active_preset['t'] if active_preset else 5
            taste_in = st.select_slider("Taste Score", options=[1, 2, 3, 4, 5], value=int(t_val), help="1 = Acrid / Off-flavor | 5 = Normal Flavor")

            p_val = active_preset['p'] if active_preset else 5
            pkg_in = st.select_slider("Packaging Seal", options=[1, 2, 3, 4, 5], value=int(p_val), help="1 = Tampered Seal | 5 = Pristine Hermetic")

        st.caption("Scores 1–2 represent organoleptic threshold breaches.")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Evaluation Action Bar
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    assess_btn = st.button("⚡ EXECUTE FORENSIC COMPLIANCE ASSAY", use_container_width=True, type="primary")

    if assess_btn or active_preset is not None:
        sample_query = {
            'food_category': chosen_cat,
            'moisture_content': moisture_input,
            'ph_level': ph_input,
            'storage_temperature': temp_input,
            'expiry_days_remaining': exp_input,
            'preservatives_used': pres_input,
            'color_score': color_in,
            'smell_score': smell_in,
            'taste_score': taste_in,
            'packaging_quality_score': pkg_in
        }

        # Predict and log
        diag = predict_single(sample_query, log_to_db=True)
        prob = diag['probability']
        status = diag['predicted_status']
        risk_tier = diag['risk_band']
        db_id = diag.get('database_log_id', 'Logged')

        # ---------------------------------------------------------------------
        # Unique Verdict Cockpit & Speedometer Display
        # ---------------------------------------------------------------------
        st.markdown(f"""
        <div class="verdict-cockpit">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 14px;">
                <div>
                    <span style="font-size: 0.80rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.08em;">
                        FORENSIC TRIAGE VERDICT · CONSIGNMENT AUDIT #{db_id}
                    </span>
                    <div style="margin-top: 4px;">
                        {'<span class="glow-safe">🛡️ CONFORMING GRADE: SAFE FOR COMMERCE</span>' if status == 'Safe' else '<span class="glow-adulterated">🚨 CONTAMINATED GRADE: ADULTERATED</span>'}
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.78rem; color: #94a3b8;">ASSIGNED RISK TIER:</span><br>
                    {'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; font-weight: 800; padding: 4px 12px; border-radius: 6px;">🟢 LOW RISK (< 0.35)</span>' if risk_tier == 'Low Risk' else ('<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; font-weight: 800; padding: 4px 12px; border-radius: 6px;">🟡 MEDIUM RISK (0.35 - 0.65)</span>' if risk_tier == 'Medium Risk' else '<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; font-weight: 800; padding: 4px 12px; border-radius: 6px;">🔴 HIGH RISK (> 0.65)</span>')}
                </div>
            </div>
        """, unsafe_allow_html=True)

        res_c1, res_c2 = st.columns([1, 1.8])

        with res_c1:
            # SVG Radial Gauge Speedometer
            gauge_stroke = "#10b981" if prob < 0.35 else ("#f59e0b" if prob <= 0.65 else "#ef4444")
            dash_offset = 283 - (prob * 283)

            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <svg width="180" height="180" viewBox="0 0 100 100" class="speedometer-svg">
                    <circle cx="50" cy="50" r="42" fill="none" stroke="#1e293b" stroke-width="8" />
                    <circle cx="50" cy="50" r="42" fill="none" stroke="{gauge_stroke}" stroke-width="8"
                            stroke-dasharray="283" stroke-dashoffset="{dash_offset}"
                            stroke-linecap="round" transform="rotate(-90 50 50)" />
                    <text x="50" y="47" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="16" font-weight="700" fill="#f8fafc">
                        {prob:.1%}
                    </text>
                    <text x="50" y="63" text-anchor="middle" font-family="'Plus Jakarta Sans', sans-serif" font-size="7" font-weight="600" fill="#94a3b8" letter-spacing="1">
                        RISK PROBABILITY
                    </text>
                </svg>
                <div style="font-size: 0.80rem; color: #94a3b8; margin-top: 4px;">
                    Calibrated Posterior Confidence: <strong>{prob:.4f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown("##### 📜 Statutory Directive & Enforcement Guidance")
            if risk_tier == "Low Risk":
                st.success("✅ **CLEARANCE NOTICE:** Sample passes all statutory physical, chemical, and organoleptic baseline tolerances. Consignment cleared for unrestricted commercial sale under FSSAI regulations.")
            elif risk_tier == "Medium Risk":
                st.warning("⚠️ **STATUTORY RE-INSPECTION ADVISORY:** Borderline chemical anomalies or thermal storage wear detected. Under Section 38, hold the consignment for secondary rapid enzymatic screening before market release.")
            else:
                st.error("🚨 **IMMEDIATE QUARANTINE ORDER:** Critical non-conformance detected. Immediate seizure warranted under Section 47 of the Food Safety and Standards Act, 2006. Dispatch duplicate sealed forensic specimens to an accredited referral laboratory.")

            st.markdown("##### 🔍 Identified Non-Conformance Vectors")
            flags = diag.get('risk_drivers', [])
            for f in flags:
                sev = f['severity']
                css_class = "forensic-flag" if sev == "High" else ("forensic-flag-warning" if sev == "Medium" else "forensic-flag-conforming")
                icon = "🔴" if sev == "High" else ("🟡" if sev == "Medium" else "🟢")
                st.markdown(f"""
                <div class="{css_class}">
                    <span>{icon} <strong>{f['feature']}:</strong> {f['detail']}</span>
                    <span style="font-weight: 700; font-size: 0.75rem; text-transform: uppercase;">{f['status']}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MODE 2: HIGH-VOLUME CARGO AUDIT
# =============================================================================
elif menu_choice == "🚢 High-Volume Cargo Audit":
    st.markdown("### 🚢 Bulk Cargo & Market Surveillance Scanner")
    st.markdown("Automated batch triage for bulk agricultural shipments and regional retail surveillance.")

    b_col1, b_col2 = st.columns([0.35, 0.65])

    with b_col1:
        st.markdown("#### 1. Download Manifest Specification")
        st.caption("Ensure your surveillance CSV contains the 10 required parameter headers.")

        template_df = pd.DataFrame([
            {'food_category': 'milk', 'moisture_content': 87.8, 'ph_level': 6.65, 'storage_temperature': 4.0, 'expiry_days_remaining': 10, 'preservatives_used': 'none', 'color_score': 5, 'smell_score': 5, 'taste_score': 5, 'packaging_quality_score': 5},
            {'food_category': 'milk', 'moisture_content': 93.5, 'ph_level': 7.90, 'storage_temperature': 14.0, 'expiry_days_remaining': -2, 'preservatives_used': 'non-permitted', 'color_score': 2, 'smell_score': 2, 'taste_score': 1, 'packaging_quality_score': 2},
            {'food_category': 'oil', 'moisture_content': 0.85, 'ph_level': 7.10, 'storage_temperature': 35.0, 'expiry_days_remaining': -5, 'preservatives_used': 'non-permitted', 'color_score': 2, 'smell_score': 2, 'taste_score': 1, 'packaging_quality_score': 2},
            {'food_category': 'honey', 'moisture_content': 24.5, 'ph_level': 5.20, 'storage_temperature': 28.0, 'expiry_days_remaining': 90, 'preservatives_used': 'none', 'color_score': 3, 'smell_score': 3, 'taste_score': 2, 'packaging_quality_score': 4}
        ])

        st.download_button(
            "📥 Download Surveillance CSV Template",
            data=template_df.to_csv(index=False),
            file_name="food_manifest_surveillance_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    with b_col2:
        st.markdown("#### 2. Ingest Manifest File")
        uploaded_csv = st.file_uploader("Upload CSV Cargo Manifest", type=['csv'])

    if uploaded_csv is not None:
        try:
            in_df = pd.read_csv(uploaded_csv)
            st.info(f"Loaded **{len(in_df)}** records. Commencing batch vectorized inference...")

            with st.spinner("Executing pipeline scoring and writing audit logs..."):
                scored_manifest = predict_batch(in_df, log_to_db=True)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Cargo Samples", len(scored_manifest))
            k2.metric("Safe Cleared", int((scored_manifest['predicted_label'] == 0).sum()))
            k3.metric("Adulterated Impounded", int((scored_manifest['predicted_label'] == 1).sum()))
            k4.metric("Adulteration Prevalence", f"{(scored_manifest['predicted_label'] == 1).mean():.1%}")

            st.markdown("##### Scored Consignment Manifest")
            st.dataframe(
                scored_manifest[['food_category', 'moisture_content', 'ph_level', 'storage_temperature', 'predicted_status', 'probability', 'risk_band']],
                use_container_width=True
            )

            out_csv = scored_manifest.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Export Scored Cargo Triage Report (CSV)",
                data=out_csv,
                file_name="scored_cargo_inspection_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Batch Processing Error: {e}")


# =============================================================================
# MODE 3: EMPIRICAL SURVEILLANCE INSIGHTS (EDA)
# =============================================================================
elif menu_choice == "📊 Empirical Surveillance Insights":
    st.markdown("### 📊 Empirical Surveillance Insights & Parameter Assays")
    st.markdown("Publication-grade statistical visualizations demonstrating parameter distributions, commodity baselines, and adulterant physics.")

    fig_map = {
        "Figure 1: Target Class Balance (Safe vs Adulterated)": "fig1_class_balance.png",
        "Figure 2: Probability Density of pH by Adulteration Class": "fig2_ph_distribution_by_class.png",
        "Figure 3: Moisture Content across Food Matrix": "fig3_moisture_by_food_category.png",
        "Figure 4: Inter-Parameter Correlation Matrix Heatmap": "fig4_correlation_heatmap.png",
        "Figure 5: Sensory & Packaging Degradation Boxplots": "fig5_sensory_score_boxplots.png",
        "Figure 6: Adulteration Prevalence Rate across Categories": "fig6_adulteration_rate_by_category.png",
        "Figure 7: Storage Temperature vs Continuous Risk": "fig7_storage_temperature_vs_risk.png",
        "Figure 8: Expiry Days Remaining by Adulteration Status": "fig8_expiry_vs_risk.png"
    }

    assay_choice = st.selectbox("Select Visual Assay to Examine", list(fig_map.keys()))
    assay_file = fig_map[assay_choice]
    assay_path = os.path.join(PROJECT_ROOT, "reports", "figures", assay_file)

    if os.path.exists(assay_path):
        st.image(assay_path, use_container_width=True)

        interp_text = {
            "fig1_class_balance.png": "Figure 1 illustrates the empirical class distribution of the primary target is_adulterated across the 6,000 synthesized samples. Conforming safe food samples constitute 3,983 records (66.38%), whereas adulterated/contaminated samples represent 2,017 records (33.62%). This 2:1 majority-to-minority ratio closely models real-world food safety surveillance scenarios where adulterated commodities remain a minority. This intentional class imbalance serves as the direct mathematical motivation for applying SMOTE oversampling exclusively on the training partition in Stage 2 to prevent algorithmic bias without incurring synthetic test leakage.",
            "fig2_ph_distribution_by_class.png": "Figure 2 displays Kernel Density Estimation (KDE) curves of hydrogen ion concentration (pH) stratified by adulteration status. Safe samples exhibit sharp, well-defined multimodal peaks centered around commodity-specific physiological equilibriums—specifically near pH 3.3 for acidic fruit juices/honey, pH 5.8 for spices/processed foods, and pH 6.65 for fresh dairy. In contrast, adulterated samples exhibit high dispersion with pronounced tails stretching into severe alkaline regions (pH > 7.5, indicative of detergent or caustic soda addition in milk) and abnormal acidic regions (pH < 4.8 in milk, indicative of bacterial souring and curdling due to cold chain breakdown).",
            "fig3_moisture_by_food_category.png": "Figure 3 demonstrates moisture distribution across all seven commodity matrices split by class status. In commodities with strictly bounded physiological water content—most notably milk (FSSAI benchmark ~87.5%) and honey (FSSAI statutory maximum 20.0%)—adulterated samples exhibit statistically significant upward shifts. In milk, dilution with extraneous water pushes moisture toward 91–96%, while in honey, inverted sugar syrups breach the 20% limit into the 22–28% range. Conversely, edible oils maintain near-zero baseline moisture (<0.2%), where any sample breaching 0.5% moisture strongly signals cheap adulterant blending or improper dehydration.",
            "fig4_correlation_heatmap.png": "The Pearson correlation heatmap in Figure 4 reveals strong, intuitive collinearities between sensory degradation indicators and regulatory risk. Specifically, taste score (r = -0.94), smell score (r = -0.90), color score (r = -0.90), and packaging quality (r = -0.89) exhibit strong negative correlations with adulteration status, validating that organoleptic failure is the primary frontline symptom of contamination. Shelf-life remaining (expiry_days_remaining) exhibits a robust negative correlation (r = -0.68), confirming that expired batches correlate heavily with spoilage.",
            "fig5_sensory_score_boxplots.png": "Figure 5 breaks down the four ordinal 1–5 inspection scores across safe and adulterated cohorts. Safe products consistently cluster at median scores of 5 (prime) with a 25th percentile at 4 (acceptable), demonstrating tight conformity. Adulterated commodities, however, show interquartile spans dropping sharply into the 1–3 range, with medians centered at 2 (moderate defect). Packaging integrity shows lower variance in safe products compared to high variance in adulterated stock, where counterfeit labeling, tampered seals, and swollen cans are frequent physical flags.",
            "fig6_adulteration_rate_by_category.png": "Figure 6 depicts the proportion of non-conforming samples identified within each specific food commodity sector. While baseline category distribution was weighted across sampling priority, adulteration prevalence remains uniformly realistic across all classes, ranging from ~31% to ~36%. Spices and dairy exhibit marginally higher non-conformance rates (35.4% and 34.8% respectively), mirroring historical FSSAI surveillance data where high economic value and liquid/powdered states make milk (water/starch/detergent) and powdered spices (synthetic dyes/sawdust) prime targets for commercial profit-motivated adulteration.",
            "fig7_storage_temperature_vs_risk.png": "Figure 7 plots storage temperature against continuous risk scores, color-coded by the three operational risk bands. For perishable chilled commodities (such as pasteurized milk and processed juices), temperatures exceeding statutory cold-chain limits (>8°C) cause a steep parabolic escalation into High Risk (>0.65). Ambient commodities (grains, spices, oils) tolerate 20–25°C, but samples stored under extreme tropical heat abuse (>35°C) exhibit accelerated lipid rancidity, pest proliferation, and chemical preservative breakdown, pushing risk scores across the 0.35 medium threshold.",
            "fig8_expiry_vs_risk.png": "Figure 8 visualizes the histogram distribution of shelf-life days remaining, highlighting the critical inflection boundary at Day 0 (statutory expiration). Samples to the left of the red dashed line (negative days remaining) are virtually 100% adulterated/non-compliant under Section 23 of the FSS Act, as selling expired stock constitutes illegal contamination regardless of organoleptic condition. Furthermore, a dense cluster of contaminated samples appears in the 0 to 10 day window, capturing near-expiry inventory dumping by vendors."
        }

        st.info(f"**Analytical Assay Assessment:**\n\n{interp_text.get(assay_file, 'Documented in Notebook 03.')}")


# =============================================================================
# MODE 4: ALGORITHMIC BENCHMARK LAB
# =============================================================================
elif menu_choice == "📈 Algorithmic Benchmark Lab":
    st.markdown("### 📈 Multi-Model Comparative Benchmarks & Decision Rules")
    st.markdown("Evaluation of all 7 machine learning models across 5-fold cross-validation and hold-out test metrics.")

    # Table
    benchmarks = [
        {"Model Engine": "Gradient Boosting (Winning Model)", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "1.0000", "5-Fold CV Recall": "0.9997 ± 0.0006"},
        {"Model Engine": "Random Forest", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9999", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Model Engine": "Decision Tree", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9998", "5-Fold CV Recall": "0.9975 ± 0.0032"},
        {"Model Engine": "Support Vector Machine (RBF)", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9985", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Model Engine": "K-Nearest Neighbors (k=7)", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9980", "5-Fold CV Recall": "0.9997 ± 0.0006"},
        {"Model Engine": "Logistic Regression", "Accuracy": "99.75%", "Precision": "99.50%", "Recall (Adulterated)": "99.75%", "F1-Score": "0.9963", "ROC-AUC": "0.9999", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Model Engine": "Gaussian Naive Bayes", "Accuracy": "99.67%", "Precision": "99.26%", "Recall (Adulterated)": "99.75%", "F1-Score": "0.9950", "ROC-AUC": "0.9993", "5-Fold CV Recall": "1.0000 ± 0.0000"}
    ]
    st.dataframe(pd.DataFrame(benchmarks), use_container_width=True)

    st.markdown("""
    > 🛡️ **Safety-First Metric Axiom:**  
    > Releasing contaminated food into the retail supply chain (Type II False Negative) triggers acute poisoning and public health crises. In contrast, holding a conforming batch for secondary lab verification (Type I False Positive) is an acceptable operational safeguard. Therefore, the winning pipeline is selected on **Recall = 100.0% (Zero False Negatives)** on hold-out evaluation.
    """)

    b_c1, b_c2 = st.columns(2)
    with b_c1:
        st.markdown("##### Overlaid ROC Curves")
        roc_img = os.path.join(PROJECT_ROOT, "reports", "figures", "fig10_model_comparison_roc.png")
        if os.path.exists(roc_img):
            st.image(roc_img, use_container_width=True)

    with b_c2:
        st.markdown("##### Confusion Matrix Comparison")
        cm_img = os.path.join(PROJECT_ROOT, "reports", "figures", "fig11_confusion_matrices.png")
        if os.path.exists(cm_img):
            st.image(cm_img, use_container_width=True)

    st.markdown("##### Global Feature Importance Ranking (Tree Ensemble)")
    feat_img = os.path.join(PROJECT_ROOT, "reports", "figures", "fig12_shap_feature_importance.png")
    if os.path.exists(feat_img):
        st.image(feat_img, use_container_width=True)

    st.markdown("---")
    st.subheader("Apriori Association Mining: High-Lift Statutory Decision Rules")
    apriori_rules = [
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
    st.table(pd.DataFrame(apriori_rules))


# =============================================================================
# MODE 5: SQLITE CHAIN-OF-CUSTODY LEDGER
# =============================================================================
elif menu_choice == "🗄️ SQLite Chain-of-Custody Ledger":
    st.markdown("### 🗄️ Statutory Chain-of-Custody Audit Ledger")
    st.markdown("Every inspection executed through FoodGuard AI is permanently recorded in SQLite `db/predictions.db` for regulatory audits.")

    all_logs = get_all_predictions_as_df()

    if len(all_logs) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Audited Samples", len(all_logs))
        c2.metric("Impounded Adulterations", int((all_logs['predicted_label'] == 1).sum()))
        c3.metric("Cleared Consignments", int((all_logs['predicted_label'] == 0).sum()))
        c4.metric("Mean Surveillance Risk", f"{all_logs['probability'].mean():.1%}")

        st.markdown("---")
        ch_c1, ch_c2 = st.columns(2)

        with ch_c1:
            st.markdown("##### Non-Conformance Rate by Commodity Matrix")
            cat_df = all_logs.groupby('food_category')['predicted_label'].mean() * 100
            st.bar_chart(cat_df)

        with ch_c2:
            st.markdown("##### Distribution of Operational Risk Bands")
            band_df = all_logs['risk_band'].value_counts()
            st.bar_chart(band_df)

        st.markdown("##### Live Audit Entries (Last 50 Events)")
        recent_log = get_recent_predictions(limit=50)
        st.dataframe(recent_log, use_container_width=True)

        full_dump = all_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Export Complete SQLite Ledger to CSV",
            data=full_dump,
            file_name=f"sqlite_statutory_audit_ledger_{len(all_logs)}_records.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No ledger entries recorded yet. Test an inspection in the Forensic Deck to populate the database.")


# =============================================================================
# MODE 6: INSTITUTIONAL CHARTER & STANDARDS
# =============================================================================
elif menu_choice == "ℹ️ Institutional Charter & Standards":
    st.markdown("### 🏛️ Academic Institutional Charter & Regulatory Citations")

    st.markdown("""
    #### 🎓 Academic Affiliation
    - **Course:** Data Mining Techniques (DMT)
    - **Degree:** Bachelor of Technology in Information Technology
    - **Institution:** K. Ramakrishnan College of Technology (Autonomous), Samayapuram, Trichy – 621112, Tamil Nadu, India
    - **Academic Year:** 2024 – 2025

    #### 👥 Project Authors
    - **Jayanth S** (Univ. Reg. No: `2403811720521023`)
    - **Jithin AK** (Univ. Reg. No: `2403811720521025`)
    - **Keerthivasan RP** (Univ. Reg. No: `2403811720521026`)

    ---

    #### 📚 Statutory Standards & Published Citations
    1. **Food Safety and Standards Act, 2006** (Act No. 34 of 2006), Ministry of Health and Family Welfare, Government of India.
    2. **FSSAI Food Safety and Standards (Food Products Standards and Food Additives) Regulations, 2011**, New Delhi.
    3. **FSSAI Manual of Methods of Analysis of Foods: Milk and Milk Products**, 2016.
    4. **FSSAI National Milk Safety Survey (NMSS) Executive Summary**, 2019.
    5. **Bureau of Indian Standards (BIS) IS 6273 (Part 1 & 2):** Guide for sensory evaluation of foods.

    ---

    #### 🔍 Known Methodological Weaknesses & Future Scope
    - **Human Sensory Limitation:** Sensory ratings (color, smell, taste, packaging) are subjective 1–5 scales carrying inter-rater variability and olfactory fatigue.
    - **IoT & Spectroscopy Enhancement:** Future iterations will interface handheld Near-Infrared (NIR) spectrometers (900–1700 nm) and Metal-Oxide Semiconductor (MOS) Electronic-Nose gas sensor arrays to replace human sensory scores with objective digitized chemometric feature vectors.
    """)
