"""
src/data_generation.py
----------------------
Domain-Grounded Synthetic Data Generation Module
Food Adulteration Risk Prediction Using Data Mining Techniques

Generates 6,000 synthetic food sample records with distributions grounded in
published Food Safety and Standards Authority of India (FSSAI) surveillance
regulations and analytical manuals.

CRITICAL DISCLAIMER:
SYNTHETIC DATA — generated for academic demonstration.
NOT a substitute for accredited laboratory testing. Do not use for real food
safety decisions.
"""

import os
import numpy as np
import pandas as pd

# Global random seed for strict academic reproducibility
RANDOM_STATE = 42

def generate_synthetic_food_data(n_samples: int = 6000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate synthetic food quality parameters reflecting real-world food adulteration
    mechanisms across seven critical food categories.

    Parameters
    ----------
    n_samples : int, default=6000
        Total number of sample observations to generate.
    random_state : int, default=42
        Seed for pseudo-random number generator.

    Returns
    -------
    pd.DataFrame
        DataFrame with 10 input features, binary target 'is_adulterated',
        continuous 'risk_score', and derived 'risk_band'.
    """
    np.random.seed(random_state)

    categories = ['milk', 'oil', 'spices', 'grains', 'honey', 'beverages', 'processed']
    # Category probabilities based on surveillance frequency
    cat_weights = [0.22, 0.16, 0.16, 0.14, 0.10, 0.11, 0.11]
    food_cat = np.random.choice(categories, size=n_samples, p=cat_weights)

    # Initialize feature arrays
    moisture = np.zeros(n_samples, dtype=float)
    ph = np.zeros(n_samples, dtype=float)
    storage_temp = np.zeros(n_samples, dtype=float)
    expiry_days = np.zeros(n_samples, dtype=int)
    preservatives = np.empty(n_samples, dtype=object)

    color_score = np.zeros(n_samples, dtype=int)
    smell_score = np.zeros(n_samples, dtype=int)
    taste_score = np.zeros(n_samples, dtype=int)
    packaging_score = np.zeros(n_samples, dtype=int)

    # Base adulteration latent intent (~35% underlying adulteration/contamination)
    adulteration_intent = np.random.binomial(1, 0.35, size=n_samples)

    for i in range(n_samples):
        cat = food_cat[i]
        is_ad = adulteration_intent[i]

        if cat == 'milk':
            # FSSAI Standards: Normal milk moisture 87.0-89.0%, pH 6.5-6.8, cold storage 2-6 C
            if is_ad:
                adulterant_type = np.random.choice(['water', 'detergent', 'starch', 'spoilage'], p=[0.4, 0.25, 0.2, 0.15])
                if adulterant_type == 'water':
                    moisture[i] = np.random.uniform(90.5, 96.0)
                    ph[i] = np.random.normal(6.6, 0.15)
                    taste_score[i] = np.random.choice([2, 3], p=[0.6, 0.4])
                    smell_score[i] = np.random.choice([3, 4], p=[0.5, 0.5])
                    color_score[i] = np.random.choice([3, 4], p=[0.6, 0.4])
                elif adulterant_type == 'detergent':
                    moisture[i] = np.random.normal(88.0, 1.2)
                    ph[i] = np.random.uniform(7.6, 8.9)  # Alkaline shift from caustic soda/detergent
                    taste_score[i] = np.random.choice([1, 2], p=[0.8, 0.2])
                    smell_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    color_score[i] = np.random.choice([2, 3], p=[0.5, 0.5])
                elif adulterant_type == 'starch':
                    moisture[i] = np.random.normal(86.5, 1.5)
                    ph[i] = np.random.normal(6.65, 0.2)
                    taste_score[i] = np.random.choice([2, 3], p=[0.7, 0.3])
                    smell_score[i] = np.random.choice([3, 4], p=[0.6, 0.4])
                    color_score[i] = np.random.choice([2, 3], p=[0.4, 0.6])
                else:  # Spoilage / temperature abuse
                    moisture[i] = np.random.normal(87.8, 1.0)
                    ph[i] = np.random.uniform(5.2, 6.2)  # Acidic curdling
                    taste_score[i] = np.random.choice([1, 2], p=[0.7, 0.3])
                    smell_score[i] = np.random.choice([1, 2], p=[0.8, 0.2])
                    color_score[i] = np.random.choice([2, 3], p=[0.6, 0.4])

                storage_temp[i] = np.random.choice([np.random.uniform(2.0, 7.0), np.random.uniform(12.0, 32.0)], p=[0.3, 0.7])
                expiry_days[i] = int(np.random.randint(-10, 4))
                preservatives[i] = np.random.choice(['none', 'permitted', 'non-permitted'], p=[0.5, 0.1, 0.4])
                packaging_score[i] = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
            else:
                moisture[i] = np.random.uniform(87.0, 89.2)
                ph[i] = np.random.normal(6.65, 0.08)
                storage_temp[i] = np.random.uniform(2.0, 6.0)
                expiry_days[i] = int(np.random.randint(2, 12))
                preservatives[i] = np.random.choice(['none', 'permitted'], p=[0.9, 0.1])
                color_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])
                smell_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])
                taste_score[i] = np.random.choice([4, 5], p=[0.15, 0.85])
                packaging_score[i] = np.random.choice([4, 5], p=[0.3, 0.7])

        elif cat == 'oil':
            # FSSAI Standards: Edible vegetable oil moisture < 0.2%, pH 5.5-6.5, ambient temp 18-30 C
            if is_ad:
                # Mineral oil / Argemone mexicana / recycled burnt oil
                moisture[i] = np.random.uniform(0.3, 2.5)
                ph[i] = np.random.choice([np.random.uniform(4.0, 5.2), np.random.uniform(6.8, 7.8)])
                storage_temp[i] = np.random.uniform(22.0, 42.0)
                expiry_days[i] = int(np.random.randint(-30, 45))
                preservatives[i] = np.random.choice(['none', 'permitted', 'non-permitted'], p=[0.4, 0.15, 0.45])
                color_score[i] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                smell_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])  # Rancidity
                taste_score[i] = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                packaging_score[i] = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.3, 0.1])
            else:
                moisture[i] = np.random.uniform(0.04, 0.18)
                ph[i] = np.random.normal(6.1, 0.2)
                storage_temp[i] = np.random.uniform(20.0, 28.0)
                expiry_days[i] = int(np.random.randint(60, 360))
                preservatives[i] = np.random.choice(['none', 'permitted'], p=[0.75, 0.25])
                color_score[i] = np.random.choice([4, 5], p=[0.3, 0.7])
                smell_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])
                taste_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])
                packaging_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])

        elif cat == 'spices':
            # FSSAI Standards: Spices moisture 6.0-12.0%, pH 5.0-6.4
            if is_ad:
                # Artificial coloring (Metanil yellow, Sudan dye), lead chromate, foreign starch/sawdust
                moisture[i] = np.random.choice([np.random.uniform(13.5, 22.0), np.random.uniform(4.0, 6.0)])
                ph[i] = np.random.choice([np.random.uniform(3.8, 4.8), np.random.uniform(6.8, 7.9)])
                storage_temp[i] = np.random.uniform(22.0, 38.0)
                expiry_days[i] = int(np.random.randint(-40, 30))
                preservatives[i] = np.random.choice(['none', 'permitted', 'non-permitted'], p=[0.3, 0.1, 0.6])
                color_score[i] = np.random.choice([1, 2], p=[0.7, 0.3])  # Unnatural artificial hue
                smell_score[i] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                taste_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
                packaging_score[i] = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            else:
                moisture[i] = np.random.uniform(7.0, 11.5)
                ph[i] = np.random.normal(5.8, 0.25)
                storage_temp[i] = np.random.uniform(18.0, 26.0)
                expiry_days[i] = int(np.random.randint(90, 400))
                preservatives[i] = np.random.choice(['none', 'permitted'], p=[0.85, 0.15])
                color_score[i] = np.random.choice([4, 5], p=[0.3, 0.7])
                smell_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])
                taste_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])
                packaging_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])

        elif cat == 'grains':
            # FSSAI Standards: Grains moisture 10.0-14.0%, pH 6.0-6.8
            if is_ad:
                # Excess moisture causing mold/mycotoxins, chalk powder, marble chips
                moisture[i] = np.random.uniform(15.0, 24.0)
                ph[i] = np.random.choice([np.random.uniform(5.0, 5.8), np.random.uniform(7.2, 8.2)])
                storage_temp[i] = np.random.uniform(24.0, 40.0)
                expiry_days[i] = int(np.random.randint(-60, 20))
                preservatives[i] = np.random.choice(['none', 'permitted', 'non-permitted'], p=[0.5, 0.1, 0.4])
                color_score[i] = np.random.choice([1, 2, 3], p=[0.45, 0.4, 0.15])
                smell_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])  # Musty mold
                taste_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                packaging_score[i] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            else:
                moisture[i] = np.random.uniform(10.5, 13.5)
                ph[i] = np.random.normal(6.4, 0.15)
                storage_temp[i] = np.random.uniform(16.0, 25.0)
                expiry_days[i] = int(np.random.randint(60, 300))
                preservatives[i] = np.random.choice(['none', 'permitted'], p=[0.9, 0.1])
                color_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])
                smell_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])
                taste_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])
                packaging_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])

        elif cat == 'honey':
            # FSSAI Standards: Honey moisture <= 20.0%, pH 3.5-4.5 (acidic)
            if is_ad:
                # Sugar syrup / Invert sugar / High fructose corn syrup (HFCS)
                moisture[i] = np.random.uniform(21.5, 29.0)  # Violates FSSAI ceiling of 20%
                ph[i] = np.random.choice([np.random.uniform(4.8, 6.2), np.random.uniform(3.0, 3.4)])
                storage_temp[i] = np.random.uniform(22.0, 38.0)
                expiry_days[i] = int(np.random.randint(-20, 60))
                preservatives[i] = np.random.choice(['none', 'permitted', 'non-permitted'], p=[0.4, 0.2, 0.4])
                color_score[i] = np.random.choice([2, 3], p=[0.6, 0.4])
                smell_score[i] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                taste_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])  # Cloying sugar taste
                packaging_score[i] = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
            else:
                moisture[i] = np.random.uniform(16.0, 19.8)
                ph[i] = np.random.normal(3.9, 0.2)
                storage_temp[i] = np.random.uniform(18.0, 26.0)
                expiry_days[i] = int(np.random.randint(120, 500))
                preservatives[i] = np.random.choice(['none', 'permitted'], p=[0.92, 0.08])
                color_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])
                smell_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])
                taste_score[i] = np.random.choice([4, 5], p=[0.15, 0.85])
                packaging_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])

        elif cat == 'beverages':
            # FSSAI Standards: Fruit juices / sodas moisture 85-94%, pH 2.8-4.2, temp 4-12 C
            if is_ad:
                # Prohibited coal tar dyes, saccharin overload, industrial water
                moisture[i] = np.random.uniform(94.5, 98.0)
                ph[i] = np.random.choice([np.random.uniform(4.5, 6.5), np.random.uniform(2.0, 2.7)])
                storage_temp[i] = np.random.uniform(12.0, 35.0)
                expiry_days[i] = int(np.random.randint(-15, 25))
                preservatives[i] = np.random.choice(['none', 'permitted', 'non-permitted'], p=[0.2, 0.2, 0.6])
                color_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                smell_score[i] = np.random.choice([1, 2, 3], p=[0.45, 0.4, 0.15])
                taste_score[i] = np.random.choice([1, 2], p=[0.7, 0.3])
                packaging_score[i] = np.random.choice([1, 2, 3], p=[0.35, 0.45, 0.2])
            else:
                moisture[i] = np.random.uniform(86.0, 93.0)
                ph[i] = np.random.normal(3.4, 0.3)
                storage_temp[i] = np.random.uniform(4.0, 10.0)
                expiry_days[i] = int(np.random.randint(30, 180))
                preservatives[i] = np.random.choice(['none', 'permitted'], p=[0.4, 0.6])
                color_score[i] = np.random.choice([4, 5], p=[0.3, 0.7])
                smell_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])
                taste_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])
                packaging_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])

        else:  # processed
            # FSSAI Standards: Bakery, snacks, ready-to-eat
            if is_ad:
                # Excess trans-fats, prohibited leavening agents, recycled frying oil
                moisture[i] = np.random.choice([np.random.uniform(25.0, 45.0), np.random.uniform(1.0, 4.0)])
                ph[i] = np.random.choice([np.random.uniform(3.5, 4.5), np.random.uniform(7.2, 8.5)])
                storage_temp[i] = np.random.uniform(22.0, 38.0)
                expiry_days[i] = int(np.random.randint(-40, 20))
                preservatives[i] = np.random.choice(['none', 'permitted', 'non-permitted'], p=[0.3, 0.15, 0.55])
                color_score[i] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                smell_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                taste_score[i] = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                packaging_score[i] = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            else:
                moisture[i] = np.random.uniform(6.0, 18.0)
                ph[i] = np.random.normal(5.5, 0.3)
                storage_temp[i] = np.random.uniform(18.0, 25.0)
                expiry_days[i] = int(np.random.randint(30, 210))
                preservatives[i] = np.random.choice(['none', 'permitted'], p=[0.6, 0.4])
                color_score[i] = np.random.choice([4, 5], p=[0.3, 0.7])
                smell_score[i] = np.random.choice([4, 5], p=[0.25, 0.75])
                taste_score[i] = np.random.choice([4, 5], p=[0.2, 0.8])
                packaging_score[i] = np.random.choice([4, 5], p=[0.3, 0.7])

    # Round floating-point features to realistic laboratory measurement precision
    moisture = np.round(moisture, 2)
    ph = np.round(ph, 2)
    storage_temp = np.round(storage_temp, 1)

    # --------------------------------------------------------------------------
    # Documented Weighted Rule for Risk Score and 'is_adulterated'
    # --------------------------------------------------------------------------
    # Base risk score starts from 0.05 (nominal baseline background risk)
    # 1. Chemical Preservative:
    #    - non-permitted: +0.28
    #    - permitted: +0.02
    # 2. Expiry Proximity:
    #    - already expired (<= 0 days): +0.25
    #    - near expiry (1-5 days): +0.10
    #    - short shelf-life (6-15 days): +0.04
    # 3. Sensory Deficit:
    #    - Normalized deficit across color, smell, taste (scale 0 to 1): weight 0.28
    # 4. Packaging Integrity Deficit:
    #    - (5 - packaging_score) / 4 * 0.10
    # 5. Physicochemical Threshold Breaches (moisture, pH, temp):
    #    - Category-specific non-conformance penalties: up to +0.25
    # 6. Gaussian measurement noise N(0, 0.05) added to simulate laboratory variability.

    score = np.full(n_samples, 0.05, dtype=float)

    # 1. Preservative penalty
    score += np.where(preservatives == 'non-permitted', 0.28, 0.0)
    score += np.where(preservatives == 'permitted', 0.02, 0.0)

    # 2. Expiry status
    score += np.where(expiry_days <= 0, 0.25, 0.0)
    score += np.where((expiry_days > 0) & (expiry_days <= 5), 0.10, 0.0)
    score += np.where((expiry_days > 5) & (expiry_days <= 15), 0.04, 0.0)

    # 3. Sensory scores deficit
    sensory_deficit = ((5 - color_score) + (5 - smell_score) + (5 - taste_score)) / 12.0
    score += sensory_deficit * 0.28

    # 4. Packaging score penalty
    score += ((5 - packaging_score) / 4.0) * 0.10

    # 5. Category-specific physicochemical deviations
    for i in range(n_samples):
        cat = food_cat[i]
        m = moisture[i]
        p = ph[i]
        t = storage_temp[i]

        if cat == 'milk':
            if m > 90.0 or m < 85.0:
                score[i] += 0.15
            if p > 7.2 or p < 6.2:
                score[i] += 0.16
            if t > 8.0:
                score[i] += 0.10
        elif cat == 'oil':
            if m > 0.22:
                score[i] += 0.18
            if p > 6.8 or p < 5.2:
                score[i] += 0.14
        elif cat == 'spices':
            if m > 12.5:
                score[i] += 0.16
            if p > 6.8 or p < 4.8:
                score[i] += 0.15
        elif cat == 'grains':
            if m > 14.2:
                score[i] += 0.16
            if p > 7.1 or p < 5.8:
                score[i] += 0.12
        elif cat == 'honey':
            if m > 20.0:  # FSSAI threshold breach
                score[i] += 0.22
            if p > 4.6 or p < 3.4:
                score[i] += 0.14
        elif cat == 'beverages':
            if m > 94.0:
                score[i] += 0.12
            if p > 4.5 or p < 2.5:
                score[i] += 0.16
            if t > 15.0:
                score[i] += 0.10
        elif cat == 'processed':
            if m > 22.0 or m < 3.0:
                score[i] += 0.14
            if p > 7.0 or p < 4.5:
                score[i] += 0.14

    # Introduce Gaussian noise N(0, 0.05) representing field inspection variance
    noise = np.random.normal(0, 0.05, size=n_samples)
    continuous_risk = np.clip(score + noise, 0.02, 0.98)

    # Primary target binary classification: threshold at 0.50
    # Realistic label overlap: samples with continuous_risk in [0.45, 0.55] have stochastic boundary
    is_adulterated = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        cr = continuous_risk[i]
        if cr >= 0.55:
            is_adulterated[i] = 1
        elif cr <= 0.42:
            is_adulterated[i] = 0
        else:
            # Overlap zone: probability proportional to risk
            p_ad = (cr - 0.42) / (0.55 - 0.42)
            is_adulterated[i] = np.random.binomial(1, p_ad)

    # Derived operational risk bands:
    # < 0.35: Low Risk (Safe)
    # 0.35 - 0.65: Medium Risk (Retest advised)
    # > 0.65: High Risk (Adulterated - lab testing recommended)
    risk_bands = []
    for r in continuous_risk:
        if r < 0.35:
            risk_bands.append('Low Risk')
        elif r <= 0.65:
            risk_bands.append('Medium Risk')
        else:
            risk_bands.append('High Risk')

    # Construct final structured DataFrame
    df = pd.DataFrame({
        'food_category': food_cat,
        'moisture_content': moisture,
        'ph_level': ph,
        'storage_temperature': storage_temp,
        'expiry_days_remaining': expiry_days,
        'preservatives_used': preservatives,
        'color_score': color_score,
        'smell_score': smell_score,
        'taste_score': taste_score,
        'packaging_quality_score': packaging_score,
        'is_adulterated': is_adulterated,
        'ground_truth_risk_score': np.round(continuous_risk, 4),
        'risk_band': risk_bands
    })

    return df

def save_dataset(output_path: str = "data/raw/food_adulteration_raw.csv") -> pd.DataFrame:
    """Generate and write raw synthetic dataset with standard metadata headers."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_food_data(n_samples=6000, random_state=RANDOM_STATE)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Synthetic dataset generated with shape {df.shape}")
    print(f"[CLASS BALANCE] Adulterated: {df['is_adulterated'].sum()} ({df['is_adulterated'].mean():.2%}), Safe: {(1 - df['is_adulterated']).sum()} ({(1 - df['is_adulterated']).mean():.2%})")
    print(f"[RISK BANDS]\n{df['risk_band'].value_counts(normalize=True).round(3)}")
    print(f"[SAVED TO] {output_path}")
    return df

if __name__ == "__main__":
    save_dataset()
