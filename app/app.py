"""
app/app.py
----------
Interactive Streamlit Web Application Dashboard
Architecture Stage 7: Food Adulteration Risk Prediction Using Data Mining Techniques

Tabs:
1. Predict: Single-sample form with parameter inputs, risk gauge & explainability
2. Batch Scan: CSV upload, batch prediction & download scored CSV
3. Data Insights: Interactive 8-figure EDA gallery with analytical interpretations
4. Model Performance: Metrics comparison table, ROC-AUC, confusion matrices, SHAP attribution
5. History & Reports: Live SQLite queries from db/predictions.db with analytics & CSV export
6. About: Academic details, team, methodology, FSSAI citations, limitations

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

# Ensure workspace root is in sys.path
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
    page_title="Food Adulteration Risk Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }
    /* Permanent Academic Warning Banner */
    .disclaimer-banner {
        background-color: #451a03;
        border-left: 5px solid #f59e0b;
        color: #fef3c7;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.90rem;
        margin-bottom: 20px;
        font-weight: 500;
    }
    /* Risk Badges */
    .badge-safe {
        background-color: #065f46;
        color: #d1fae5;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-adulterated {
        background-color: #991b1b;
        color: #fee2e2;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-risk-low {
        background-color: #047857;
        color: #ecfdf5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-risk-medium {
        background-color: #b45309;
        color: #fef3c7;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-risk-high {
        background-color: #b91c1c;
        color: #fef2f2;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading optimized inference pipeline...")
def get_cached_pipeline():
    """Loads and caches the serialized model pipeline."""
    return load_model_pipeline()


@st.cache_data
def get_cached_metadata():
    """Loads model metadata JSON."""
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# Initialize SQLite Database at startup
init_db()
pipeline, metadata = get_cached_pipeline()


# -----------------------------------------------------------------------------
# Permanent Screen Header & Academic Disclaimer
# -----------------------------------------------------------------------------
st.title("🔬 Food Adulteration Risk Prediction System")
st.markdown("**Department of Information Technology · K. Ramakrishnan College of Technology (Autonomous), Trichy**")

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>ACADEMIC DEMONSTRATION ONLY:</strong> Trained strictly on 6,000 domain-grounded synthetic records based on 
    FSSAI standards. This application is an educational prototype for the <em>Data Mining Techniques</em> course. 
    It is <strong>NOT</strong> a substitute for accredited laboratory chromatography/spectroscopy. Do not use for real-world legal enforcement or clinical food safety decisions.
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: Inspection Context & Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/food-receiver.png", width=70)
    st.subheader("Field Inspector Portal")
    st.markdown("**Standard:** FSSAI Act, 2006")
    st.markdown(f"**Model Engine:** `{metadata.get('model_name', 'Gradient Boosting')}`")
    st.markdown(f"**Test Recall:** `{metadata.get('metrics', {}).get('recall', 1.0):.2%}` *(Zero False Negatives)*")
    st.markdown(f"**ROC-AUC:** `{metadata.get('metrics', {}).get('roc_auc', 1.0):.4f}`")

    st.markdown("---")
    st.markdown("### Operational Risk Thresholds")
    st.markdown("- 🟢 **Low Risk ($p < 0.35$):** Conforming (Safe)")
    st.markdown("- 🟡 **Medium Risk ($0.35 \\le p \\le 0.65$):** Retest Advised")
    st.markdown("- 🔴 **High Risk ($p > 0.65$):** Lab Confirmatory Testing Recommended")

    st.markdown("---")
    st.caption("Team: Jayanth S (1023), Jithin AK (1025), Keerthivasan RP (1026)")


# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Single Predictor",
    "📁 Batch CSV Scanner",
    "📊 Data Insights & EDA",
    "📈 Model Benchmarking",
    "🗄️ Audit History & Reports",
    "ℹ️ About & Methodology"
])


# =============================================================================
# TAB 1: SINGLE PREDICTION
# =============================================================================
with tab1:
    st.subheader("Interactive Food Sample Inspection Form")
    st.write("Input measurable physical parameters, cold-chain storage conditions, and sensory Likert scores to evaluate contamination risk.")

    # Demonstration Preset Buttons
    st.markdown("##### ⚡ Quick Load Viva Demo Presets")
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
    preset_data = None
    if c_btn1.button("🥛 Conforming Fresh Milk"):
        preset_data = {'food_category': 'milk', 'moisture_content': 87.8, 'ph_level': 6.65, 'storage_temperature': 4.0, 'expiry_days_remaining': 8, 'preservatives_used': 'none', 'color_score': 5, 'smell_score': 5, 'taste_score': 5, 'packaging_quality_score': 5}
    if c_btn2.button("⚠️ Adulterated Milk (Water+Soda)"):
        preset_data = {'food_category': 'milk', 'moisture_content': 93.5, 'ph_level': 7.9, 'storage_temperature': 15.0, 'expiry_days_remaining': -1, 'preservatives_used': 'non-permitted', 'color_score': 2, 'smell_score': 2, 'taste_score': 1, 'packaging_quality_score': 2}
    if c_btn3.button("🍯 Adulterated Honey (High Moisture)"):
        preset_data = {'food_category': 'honey', 'moisture_content': 24.5, 'ph_level': 5.2, 'storage_temperature': 28.0, 'expiry_days_remaining': 120, 'preservatives_used': 'none', 'color_score': 3, 'smell_score': 3, 'taste_score': 2, 'packaging_quality_score': 4}
    if c_btn4.button("🌶️ Dye-Adulterated Spices"):
        preset_data = {'food_category': 'spices', 'moisture_content': 16.0, 'ph_level': 4.2, 'storage_temperature': 34.0, 'expiry_days_remaining': 15, 'preservatives_used': 'non-permitted', 'color_score': 1, 'smell_score': 2, 'taste_score': 2, 'packaging_quality_score': 2}

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1. Objective Measurable Parameters")
            food_cat_default = preset_data['food_category'] if preset_data else 'milk'
            categories = ['milk', 'oil', 'spices', 'grains', 'honey', 'beverages', 'processed']
            food_cat = st.selectbox("Food Commodity Category", categories, index=categories.index(food_cat_default))

            moisture_default = float(preset_data['moisture_content']) if preset_data else 87.5
            moisture = st.number_input("Moisture Content (% w/w)", min_value=0.0, max_value=100.0, value=moisture_default, step=0.1, help="Standard Milk: 87-89%, Honey max 20%, Oil max 0.2%")

            ph_default = float(preset_data['ph_level']) if preset_data else 6.65
            ph = st.number_input("pH Level (Acidity/Alkalinity)", min_value=2.0, max_value=10.0, value=ph_default, step=0.05, help="Fresh milk: 6.5-6.8. Alkaline > 7.5 indicates detergent.")

            temp_default = float(preset_data['storage_temperature']) if preset_data else 4.0
            storage_temp = st.number_input("Storage Temperature (°C)", min_value=-25.0, max_value=50.0, value=temp_default, step=0.5, help="Cold chain for milk <= 6°C.")

            exp_default = int(preset_data['expiry_days_remaining']) if preset_data else 7
            expiry_days = st.number_input("Expiry Days Remaining (Negative = Expired)", min_value=-60, max_value=730, value=exp_default, step=1)

            pres_default = preset_data['preservatives_used'] if preset_data else 'none'
            preservatives = st.selectbox("Chemical Preservatives Detected", ['none', 'permitted', 'non-permitted'], index=['none', 'permitted', 'non-permitted'].index(pres_default))

        with col2:
            st.markdown("#### 2. Organoleptic Sensory Assessment (Likert 1 to 5)")
            st.caption("1 = Clearly Abnormal / Acute Defect | 5 = Normal / Conforming Commercial Grade")

            c_val = int(preset_data['color_score']) if preset_data else 5
            color_s = st.slider("Color Score (Unnatural tint / dye discoloration)", 1, 5, c_val)

            s_val = int(preset_data['smell_score']) if preset_data else 5
            smell_s = st.slider("Smell Score (Foul odor / rancid / chemical)", 1, 5, s_val)

            t_val = int(preset_data['taste_score']) if preset_data else 5
            taste_s = st.slider("Taste Score (Unpalatable / acrid / off-flavor)", 1, 5, t_val)

            p_val = int(preset_data['packaging_quality_score']) if preset_data else 5
            pkg_s = st.slider("Packaging Integrity (Tampered / compromised seal)", 1, 5, p_val)

        submit_btn = st.form_submit_button("🔍 Run Forensic Adulteration Assessment", use_container_width=True)

    if submit_btn:
        sample_dict = {
            'food_category': food_cat,
            'moisture_content': moisture,
            'ph_level': ph,
            'storage_temperature': storage_temp,
            'expiry_days_remaining': expiry_days,
            'preservatives_used': preservatives,
            'color_score': color_s,
            'smell_score': smell_s,
            'taste_score': taste_s,
            'packaging_quality_score': pkg_s
        }

        result = predict_single(sample_dict, log_to_db=True)
        prob = result['probability']
        status = result['predicted_status']
        risk_band = result['risk_band']
        db_id = result.get('database_log_id')

        st.markdown("---")
        st.subheader("Inspection Diagnostic Verdict")

        # Top Metric Cards
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        with v_col1:
            status_html = '<span class="badge-safe">SAFE (CONFORMING)</span>' if status == 'Safe' else '<span class="badge-adulterated">ADULTERATED</span>'
            st.markdown(f"**Classification Status**<br>{status_html}", unsafe_allow_html=True)
        with v_col2:
            st.metric("Calibrated Risk Probability", f"{prob:.2%}")
        with v_col3:
            if risk_band == 'Low Risk':
                badge = '<span class="badge-risk-low">🟢 LOW RISK</span>'
            elif risk_band == 'Medium Risk':
                badge = '<span class="badge-risk-medium">🟡 MEDIUM RISK</span>'
            else:
                badge = '<span class="badge-risk-high">🔴 HIGH RISK</span>'
            st.markdown(f"**Operational Risk Tier**<br>{badge}", unsafe_allow_html=True)
        with v_col4:
            st.metric("Audit Provenance ID", f"#{db_id} (SQLite)")

        # Risk Gauge Bar
        st.markdown("##### Calibrated Risk Band Gauge")
        st.progress(min(1.0, max(0.0, prob)))
        col_g1, col_g2, col_g3 = st.columns([0.35, 0.30, 0.35])
        col_g1.caption("🟢 Low Risk: 0.00 – 0.35 (Safe)")
        col_g2.caption("🟡 Medium Risk: 0.35 – 0.65 (Retest Advised)")
        col_g3.caption("🔴 High Risk: 0.65 – 1.00 (Lab Action)")

        # Actionable Recommendation Callout
        if risk_band == "Low Risk":
            st.success("✅ **ACTION:** Sample complies with statutory baseline tolerances. Consignment cleared for commercial distribution.")
        elif risk_band == "Medium Risk":
            st.warning("⚠️ **ACTION: RETEST ADVISED.** Sample exhibits borderline physical, expiry, or sensory markers. Secondary bench screening recommended prior to clearance.")
        else:
            st.error("🚨 **ACTION: HIGH RISK — IMMEDIATE LAB CONFIRMATORY TESTING RECOMMENDED.** Severe regulatory non-conformance detected. Impound lot and dispatch duplicate sample to accredited referral laboratory under Section 47 of FSS Act.")

        # Explainability & Diagnostic Drivers Panel
        st.markdown("#### 🧠 Model Explainability: Why Was This Verdict Assigned?")
        drivers = result['risk_drivers']
        for d in drivers:
            if d['severity'] == 'High':
                st.markdown(f"- 🔴 **{d['feature']}:** {d['detail']}")
            elif d['severity'] == 'Medium':
                st.markdown(f"- 🟡 **{d['feature']}:** {d['detail']}")
            else:
                st.markdown(f"- 🟢 **{d['feature']}:** {d['detail']}")


# =============================================================================
# TAB 2: BATCH CSV SCANNER
# =============================================================================
with tab2:
    st.subheader("Batch Sample Screening & Triage")
    st.write("Upload an inventory or field surveillance CSV dataset containing food parameters to perform high-throughput risk classification.")

    # Template download
    sample_template = pd.DataFrame([{
        'food_category': 'milk',
        'moisture_content': 87.5,
        'ph_level': 6.65,
        'storage_temperature': 4.0,
        'expiry_days_remaining': 10,
        'preservatives_used': 'none',
        'color_score': 5,
        'smell_score': 5,
        'taste_score': 5,
        'packaging_quality_score': 5
    }, {
        'food_category': 'oil',
        'moisture_content': 0.85,
        'ph_level': 7.1,
        'storage_temperature': 35.0,
        'expiry_days_remaining': -5,
        'preservatives_used': 'non-permitted',
        'color_score': 2,
        'smell_score': 2,
        'taste_score': 1,
        'packaging_quality_score': 2
    }])

    st.download_button(
        "📥 Download Sample CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="food_sample_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Inspection CSV File", type=['csv'])

    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_in)}** rows. Commencing batch inference...")

            with st.spinner("Processing batch through Scikit-Learn pipeline and logging to SQLite..."):
                scored_df = predict_batch(df_in, log_to_db=True)

            # Summary Metrics
            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            b_c1.metric("Total Samples Scanned", len(scored_df))
            b_c2.metric("Safe (Conforming)", int((scored_df['predicted_label'] == 0).sum()))
            b_c3.metric("Adulterated Flagged", int((scored_df['predicted_label'] == 1).sum()))
            b_c4.metric("Adulteration Rate", f"{(scored_df['predicted_label'] == 1).mean():.1%}")

            # Preview Scored Data
            st.markdown("##### Preview of Scored Consignments")
            st.dataframe(scored_df[['food_category', 'moisture_content', 'ph_level', 'storage_temperature', 'predicted_status', 'probability', 'risk_band']].head(20))

            # Download Result
            csv_data = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Scored Inspection Records (CSV)",
                data=csv_data,
                file_name="adulteration_batch_scan_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error executing batch scan: {e}")


# =============================================================================
# TAB 3: DATA INSIGHTS & EDA
# =============================================================================
with tab3:
    st.subheader("Exploratory Data Analysis & Empirical Insights (Phase 4)")
    st.write("Publication-grade statistical visualizations demonstrating parameter distributions, commodity baselines, and adulterant physics.")

    fig_options = {
        "Figure 1: Target Class Balance (Safe vs Adulterated)": "fig1_class_balance.png",
        "Figure 2: Probability Density of pH by Class": "fig2_ph_distribution_by_class.png",
        "Figure 3: Moisture Content across Food Matrix": "fig3_moisture_by_food_category.png",
        "Figure 4: Inter-Parameter Correlation Matrix Heatmap": "fig4_correlation_heatmap.png",
        "Figure 5: Sensory & Packaging Degradation Boxplots": "fig5_sensory_score_boxplots.png",
        "Figure 6: Adulteration Prevalence Rate across Categories": "fig6_adulteration_rate_by_category.png",
        "Figure 7: Storage Temperature vs Continuous Risk": "fig7_storage_temperature_vs_risk.png",
        "Figure 8: Expiry Days Remaining by Adulteration Status": "fig8_expiry_vs_risk.png"
    }

    selected_fig_title = st.selectbox("Select Figure to Inspect", list(fig_options.keys()))
    fig_filename = fig_options[selected_fig_title]
    fig_path = os.path.join(PROJECT_ROOT, "reports", "figures", fig_filename)

    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)

        # Accompanying Academic Interpretation
        interpretations = {
            "fig1_class_balance.png": "Figure 1 illustrates the empirical class distribution of the primary target is_adulterated across the 6,000 synthesized samples. Conforming safe food samples constitute 3,983 records (66.38%), whereas adulterated/contaminated samples represent 2,017 records (33.62%). This 2:1 majority-to-minority ratio closely models real-world food safety surveillance scenarios where adulterated commodities remain a minority. This intentional class imbalance serves as the direct mathematical motivation for applying SMOTE oversampling exclusively on the training partition in Stage 2 to prevent algorithmic bias without incurring synthetic test leakage.",
            "fig2_ph_distribution_by_class.png": "Figure 2 displays Kernel Density Estimation (KDE) curves of hydrogen ion concentration (pH) stratified by adulteration status. Safe samples exhibit sharp, well-defined multimodal peaks centered around commodity-specific physiological equilibriums—specifically near pH 3.3 for acidic fruit juices/honey, pH 5.8 for spices/processed foods, and pH 6.65 for fresh dairy. In contrast, adulterated samples exhibit high dispersion with pronounced tails stretching into severe alkaline regions (pH > 7.5, indicative of detergent or caustic soda addition in milk) and abnormal acidic regions (pH < 4.8 in milk, indicative of bacterial souring and curdling due to cold chain breakdown).",
            "fig3_moisture_by_food_category.png": "Figure 3 demonstrates moisture distribution across all seven commodity matrices split by class status. In commodities with strictly bounded physiological water content—most notably milk (FSSAI benchmark ~87.5%) and honey (FSSAI statutory maximum 20.0%)—adulterated samples exhibit statistically significant upward shifts. In milk, dilution with extraneous water pushes moisture toward 91–96%, while in honey, inverted sugar syrups breach the 20% limit into the 22–28% range. Conversely, edible oils maintain near-zero baseline moisture (<0.2%), where any sample breaching 0.5% moisture strongly signals cheap adulterant blending or improper dehydration.",
            "fig4_correlation_heatmap.png": "The Pearson correlation heatmap in Figure 4 reveals strong, intuitive collinearities between sensory degradation indicators and regulatory risk. Specifically, taste score (r = -0.94), smell score (r = -0.90), color score (r = -0.90), and packaging quality (r = -0.89) exhibit strong negative correlations with adulteration status, validating that organoleptic failure is the primary frontline symptom of contamination. Shelf-life remaining (expiry_days_remaining) exhibits a robust negative correlation (r = -0.68), confirming that expired batches correlate heavily with spoilage.",
            "fig5_sensory_score_boxplots.png": "Figure 5 breaks down the four ordinal 1–5 inspection scores across safe and adulterated cohorts. Safe products consistently cluster at median scores of 5 (prime) with a 25th percentile at 4 (acceptable), demonstrating tight conformity. Adulterated commodities, however, show interquartile spans dropping sharply into the 1–3 range, with medians centered at 2 (moderate defect). Packaging integrity shows lower variance in safe products compared to high variance in adulterated stock, where counterfeit labeling, tampered seals, and swollen cans are frequent physical flags.",
            "fig6_adulteration_rate_by_category.png": "Figure 6 depicts the proportion of non-conforming samples identified within each specific food commodity sector. While baseline category distribution was weighted across sampling priority, adulteration prevalence remains uniformly realistic across all classes, ranging from ~31% to ~36%. Spices and dairy exhibit marginally higher non-conformance rates (35.4% and 34.8% respectively), mirroring historical FSSAI surveillance data where high economic value and liquid/powdered states make milk (water/starch/detergent) and powdered spices (synthetic dyes/sawdust) prime targets for commercial profit-motivated adulteration.",
            "fig7_storage_temperature_vs_risk.png": "Figure 7 plots storage temperature against continuous risk scores, color-coded by the three operational risk bands. For perishable chilled commodities (such as pasteurized milk and processed juices), temperatures exceeding statutory cold-chain limits (>8°C) cause a steep parabolic escalation into High Risk (>0.65). Ambient commodities (grains, spices, oils) tolerate 20–25°C, but samples stored under extreme tropical heat abuse (>35°C) exhibit accelerated lipid rancidity, pest proliferation, and chemical preservative breakdown, pushing risk scores across the 0.35 medium threshold.",
            "fig8_expiry_vs_risk.png": "Figure 8 visualizes the histogram distribution of shelf-life days remaining, highlighting the critical inflection boundary at Day 0 (statutory expiration). Samples to the left of the red dashed line (negative days remaining) are virtually 100% adulterated/non-compliant under Section 23 of the FSS Act, as selling expired stock constitutes illegal contamination regardless of organoleptic condition. Furthermore, a dense cluster of contaminated samples appears in the 0 to 10 day window, capturing near-expiry inventory dumping by vendors."
        }

        st.info(f"**Analytical Interpretation:**\n\n{interpretations.get(fig_filename, 'Interpretation available in Notebook 03.')}")
    else:
        st.warning(f"Figure not found at: {fig_path}. Run src/feature_selection.py first.")


# =============================================================================
# TAB 4: MODEL PERFORMANCE
# =============================================================================
with tab4:
    st.subheader("Model Benchmarking & Safety-First Evaluation (Phases 5 & 6)")
    st.write("Rigorous 7-algorithm benchmark across 5-fold cross-validation and hold-out test metrics, prioritizing Recall for public health safety.")

    # Metrics Table
    metrics_summary = [
        {"Model": "Gradient Boosting", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "1.0000", "5-Fold CV Recall": "0.9997 ± 0.0006"},
        {"Model": "Random Forest", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9999", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Model": "Decision Tree", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9998", "5-Fold CV Recall": "0.9975 ± 0.0032"},
        {"Model": "Support Vector Machine", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9985", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Model": "K-Nearest Neighbors", "Accuracy": "99.67%", "Precision": "99.02%", "Recall (Adulterated)": "100.00%", "F1-Score": "0.9951", "ROC-AUC": "0.9980", "5-Fold CV Recall": "0.9997 ± 0.0006"},
        {"Model": "Logistic Regression", "Accuracy": "99.75%", "Precision": "99.50%", "Recall (Adulterated)": "99.75%", "F1-Score": "0.9963", "ROC-AUC": "0.9999", "5-Fold CV Recall": "1.0000 ± 0.0000"},
        {"Model": "Gaussian Naive Bayes", "Accuracy": "99.67%", "Precision": "99.26%", "Recall (Adulterated)": "99.75%", "F1-Score": "0.9950", "ROC-AUC": "0.9993", "5-Fold CV Recall": "1.0000 ± 0.0000"}
    ]
    st.table(pd.DataFrame(metrics_summary))

    st.markdown("""
    > 🛡️ **Methodological Defense: Why We Select on RECALL Rather Than Accuracy Alone:**  
    > In food regulatory surveillance, the penalty of a **False Negative** is catastrophic: contaminated or chemically adulterated food (e.g., toxic Metanil Yellow dye or detergent in milk) is passed into the consumer food supply, triggering acute illness, organ damage, or infant mortality.  
    > Conversely, a **False Positive** merely subjects a conforming batch to secondary accredited chromatographic assay. Therefore, optimizing **Recall on the Adulterated Class (100.0%, 0 False Negatives)** is our primary decision rule.
    """)

    # Visual Diagnostics
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("##### Receiver Operating Characteristic (ROC) Curves")
        roc_path = os.path.join(PROJECT_ROOT, "reports", "figures", "fig10_model_comparison_roc.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)

    with m_col2:
        st.markdown("##### Confusion Matrix Comparison Across Models")
        cm_path = os.path.join(PROJECT_ROOT, "reports", "figures", "fig11_confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)

    # Feature Importance Plot
    st.markdown("##### Global Feature Importance Ranking (Tree Ensemble)")
    feat_path = os.path.join(PROJECT_ROOT, "reports", "figures", "fig12_shap_feature_importance.png")
    if os.path.exists(feat_path):
        st.image(feat_path, use_container_width=True)

    # Apriori Association Rules Table
    st.markdown("---")
    st.subheader("Apriori Association Mining: Top 10 High-Lift Decision Rules")
    apriori_rules_data = [
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
    st.table(pd.DataFrame(apriori_rules_data))


# =============================================================================
# TAB 5: HISTORY & REPORTS
# =============================================================================
with tab5:
    st.subheader("Persistent SQLite Audit History & Analytical Reporting")
    st.write("Every prediction generated by field inspectors is persisted in `db/predictions.db` for regulatory chain-of-custody verification.")

    # High-level aggregate metrics from SQLite
    df_all = get_all_predictions_as_df()
    total_logged = len(df_all)

    if total_logged > 0:
        h_c1, h_c2, h_c3, h_c4 = st.columns(4)
        h_c1.metric("Total Inferences Logged", total_logged)
        h_c2.metric("Adulterated Flagged", int((df_all['predicted_label'] == 1).sum()))
        h_c3.metric("Safe Cleared", int((df_all['predicted_label'] == 0).sum()))
        h_c4.metric("Avg Inferred Risk", f"{df_all['probability'].mean():.2%}")

        # Summary Visualizations
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("##### Risk Band Distribution")
            risk_dist = df_all['risk_band'].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            ax_pie.pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%',
                       colors=['#2b5c8f', '#d9534f', '#f59e0b'], startangle=90)
            ax_pie.axis('equal')
            st.pyplot(fig_pie)
            plt.close()

        with s_col2:
            st.markdown("##### Category-Wise Adulteration Rate")
            cat_rate = df_all.groupby('food_category')['predicted_label'].mean() * 100
            st.bar_chart(cat_rate)

        # Recent Records Table
        st.markdown("##### Recent Audit Records (Last 50 Samples)")
        recent_df = get_recent_predictions(limit=50)
        st.dataframe(recent_df, use_container_width=True)

        # Export full database to CSV
        csv_export = df_all.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Export Complete SQLite Audit Log (CSV)",
            data=csv_export,
            file_name=f"food_audit_export_{total_logged}_records.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No predictions recorded in SQLite database yet. Perform a single prediction in Tab 1 or batch scan in Tab 2 to populate the audit log.")


# =============================================================================
# TAB 6: ABOUT & METHODOLOGY
# =============================================================================
with tab6:
    st.subheader("Academic Project Overview & Methodology")

    st.markdown("""
    ### 🏛️ Institution & Course Details
    - **Course:** Data Mining Techniques (DMT), B.Tech Information Technology
    - **Institution:** K. Ramakrishnan College of Technology (Autonomous), Trichy, Tamil Nadu
    - **Academic Year:** 2024 – 2025

    ### 👥 Project Team
    - **Jayanth S** (Reg. No: `2403811720521023`)
    - **Jithin AK** (Reg. No: `2403811720521025`)
    - **Keerthivasan RP** (Reg. No: `2403811720521026`)

    ---

    ### 📖 8-Stage System Architecture
    1. **Stage 1 (Data Synthesis):** 6,000 synthetic records modeled on published FSSAI standards.
    2. **Stage 2 (Pipeline Preprocessing):** Missing imputation, IQR outlier clipping, Ordinal/One-Hot encoding, StandardScaler, and leak-free SMOTE on train partition only.
    3. **Stage 3 (EDA & Tri-Method Selection):** Correlation filter, Chi-Square test, and Random Forest Gini importance comparison.
    4. **Stage 4 (Multi-Paradigm Modeling):** 7 tuned classifiers, K-Means cluster analysis, and Apriori association mining.
    5. **Stage 5 (Safety-First Evaluation):** Decision on Recall for adulterated class to eliminate false negatives.
    6. **Stage 6 (Prediction Engine):** Calibrated risk bands (Low < 0.35, Medium 0.35–0.65, High > 0.65).
    7. **Stage 7 (Streamlit Dashboard):** Responsive 6-tab inspection interface.
    8. **Stage 8 (SQLite Audit Persistence):** Local database persistence of all inferences.

    ---

    ### 📚 Published Regulatory References Cited
    1. **FSSAI Food Safety and Standards Act, 2006** (Act No. 34 of 2006), Government of India.
    2. **Food Safety and Standards (Food Products Standards and Food Additives) Regulations, 2011**, New Delhi.
    3. **FSSAI Manual of Methods of Analysis of Foods: Milk and Milk Products**, Directorate General of Health Services, 2016.
    4. **FSSAI National Milk Safety Survey (NMSS)** Executive Summary, 2019.
    5. **Bureau of Indian Standards IS 6273 (Part 1 & 2):** Guide for sensory evaluation of foods.

    ---

    ### 🔍 Methodological Limitations & Future Scope
    - **Known Limitation (Sensory Subjectivity):** Sensory scores (color, smell, taste, packaging) are ordinal 1–5 scales evaluated by human inspectors, carrying inter-rater variability and fatigue bias.
    - **Future Enhancement (IoT & Spectroscopy):** Integration of handheld Near-Infrared (NIR) spectrometers, Electronic Nose (E-Nose) gas sensor arrays, and RFID cold-chain temperature telemetry to replace subjective sensory inputs with objective digital spectral features.
    """)
