# Food Adulteration Risk Prediction Using Data Mining Techniques

## Academic Project Report
**Course:** Data Mining Techniques (DMT)  
**Degree:** Bachelor of Technology in Information Technology  
**Institution:** K. Ramakrishnan College of Technology (Autonomous), Kariyamanickam Road, Samayapuram, Tiruchirappalli – 621112, Tamil Nadu, India  
**Academic Year:** 2024 – 2025  

### Submitted By:
1. **Jayanth S** — University Register No: `2403811720521023`
2. **Jithin AK** — University Register No: `2403811720521025`
3. **Keerthivasan RP** — University Register No: `2403811720521026`

---

> ⚠️ **MANDATORY ACADEMIC DISCLAIMER: SYNTHETIC DATA STATEMENT**  
> *This research project and its software artifacts utilize a domain-modeled synthetic dataset of 6,000 observations mathematically generated to mirror published Food Safety and Standards Authority of India (FSSAI) regulatory standards. The findings, metrics, and models presented herein are engineered strictly for academic evaluation and pedagogical demonstration in data mining. They do not constitute certified empirical laboratory findings and must not be utilized for clinical diagnosis, commercial certification, or statutory legal enforcement.*

---

## Abstract

Food adulteration represents a severe and pervasive threat to global public health, consumer economic rights, and international supply chain security. Conventional food quality enforcement relies heavily on laboratory wet-chemistry assays (such as gas chromatography and mass spectrometry) which, while highly definitive, are cost-prohibitive, time-intensive, and inherently centralized, creating severe inspection backlogs at regional checkpoints. 

This project establishes a comprehensive, reproducible, end-to-end data mining and machine learning framework designed to predict food adulteration risk from measurable physical, chemical, logistical, and organoleptic sensory parameters. Addressing the absence of public, sample-level regulatory datasets, we engineered a 6,000-sample dataset across seven vital food matrices (*Milk, Edible Oil, Spices, Grains, Honey, Beverages, and Processed Foods*), strictly parameterized against published FSSAI standards. A robust, leak-free preprocessing pipeline was implemented featuring Interquartile Range (IQR) outlier clipping, `ColumnTransformer` feature scaling, and Synthetic Minority Over-sampling Technique (SMOTE) applied strictly to the training partition to prevent catastrophic data leakage. 

We benchmarked seven supervised classification algorithms (Decision Tree, Random Forest, Gaussian Naive Bayes, Support Vector Machine, Logistic Regression, K-Nearest Neighbors, and Gradient Boosting) optimized via 5-fold cross-validation. Emphasizing a safety-first objective, the winning model was selected on **Recall for the adulterated class (100.0% test recall, 0 False Negatives, ROC-AUC = 1.0000)**, recognizing that releasing contaminated food into commerce carries severe health hazards, whereas false positives merely trigger secondary laboratory assays. Unsupervised K-Means clustering ($k=2$, Silhouette Score = 0.4131) empirically validated that contaminated samples form natural, unguided multivariate clusters (99.6% purity), while Apriori association mining discovered high-lift decision heuristics (e.g., unauthorized preservatives yielding Lift = 2.97). The production pipeline was serialized into an interactive multi-tab Streamlit dashboard backed by an audit-grade SQLite database (`db/predictions.db`) for regulatory chain-of-custody tracking. Finally, we analyze the methodological limitations of human sensory subjectivity and outline future integration with Near-Infrared (NIR) spectroscopy and Electronic-Nose IoT sensor arrays.

---

## 1. Introduction

### 1.1 Background & Public Health Relevance
Food adulteration is defined under Section 3(1)(a) of the Indian Food Safety and Standards Act (FSS Act), 2006, as the intentional debasement of food quality through the addition of inferior, foreign, or toxic substances, or the extraction of vital nutritional constituents for commercial gain. Economically Motivated Adulteration (EMA) impacts high-turnover consumer commodities:
- **Dairy Products:** Dilution of raw milk with untreated water, disguised by adding neutralizers (caustic soda, detergents) to elevate specific gravity and prevent curdling, and starches/maltodextrin to mask reduced solids-not-fat (SNF).
- **Spices & Condiments:** Blending carcinogenic coal-tar dyes (e.g. Metanil Yellow, Sudan dyes) or toxic minerals (lead chromate) into turmeric and chili powders to impart artificial vibrancy.
- **Edible Oils:** Adulteration of premium mustard and vegetable oils with cheap mineral oils or toxic Argemone mexicana oil, leading to fatal epidemic dropsy.
- **Honey:** Dilution with industrial high-fructose corn syrups (HFCS) and inverted sugar syrups that evade standard carbohydrate tests.

The public health consequences range from acute gastrointestinal toxicity, heavy metal poisoning, and metabolic disruption to chronic hepatic failure, renal damage, and carcinogenesis.

### 1.2 Problem Statement
While the Food Safety and Standards Authority of India (FSSAI) executes periodic national surveillance (e.g., the National Milk Safety Survey), enforcement agencies face major operational constraints:
1. **Inspection Lag:** State food safety officers cannot dispatch every suspected consignment to central referral laboratories due to budget constraints and turnaround times of 7 to 21 days.
2. **Binary Decision Dilemma:** Inspectors require a triage mechanism to separate clearly safe commodities from borderline and high-risk consignments.
3. **Data Availability:** Aggregate compliance reports are published annually, but granular sample-level datasets containing multi-parametric quality attributes remain confidential or proprietary.

### 1.3 Project Objectives
The objective of this academic research is to build an end-to-end data mining system that:
1. Generates a realistic, statistically grounded 6,000-sample synthetic dataset across 7 commodity sectors reflecting FSSAI physical and chemical tolerances.
2. Formulates an automated Scikit-Learn preprocessing pipeline preventing SMOTE data leakage.
3. Conducts multi-method exploratory feature selection comparing Pearson correlation, Chi-Square independence, and Random Forest Gini importance.
4. Trains, tunes, and evaluates 7 classification models, K-Means clustering, and Apriori association rule mining.
5. Implements a safety-critical model selection protocol prioritizing **Recall** over raw accuracy.
6. Packages the winning model into a publicly deployable Streamlit inspection dashboard with SQLite database provenance logging.

---

## 2. Literature Survey

Data mining and machine learning have increasingly been recognized as pivotal tools for automated food fraud detection, quality verification, and chemometrics. Below is an academic review of five real, peer-reviewed studies informing our research architecture:

1. **Ellis, D. I., Brewster, V. L., Dunn, W. B., Allwood, J. W., & Goodacre, R. (2012).**  
   *Title:* "Point-and-shoot: rapid qualitative and quantitative detection of food adulteration using spectroscopic and sensor technologies."  
   *Journal:* *Chemical Society Reviews*, 41(2), 570–585.  
   *Contribution:* This landmark paper reviews rapid, non-destructive analytical techniques for food authentication. The authors emphasize that chemometric algorithms coupled with sensory arrays can intercept adulteration at field borders, establishing the foundation for rapid non-laboratory triage.

2. **Callao, M. P., & Ruisánchez, I. (2018).**  
   *Title:* "An overview of multivariate qualitative methods for food fraud detection."  
   *Journal:* *Food Control*, 86, 318–324.  
   *Contribution:* Callao and Ruisánchez examine multi-class classification and one-class discrimination techniques (SIMCA, PLS-DA, SVM) in food authentication. They emphasize that food matrices exhibit non-linear interactions requiring kernelized or tree-based classifiers to separate complex adulterant blends.

3. **Nascimento, C. F., Santos, P. M., Pereira-Filho, E. R., & Rocha, F. R. (2017).**  
   *Title:* "Recent advances in infrared spectroscopy and chemometrics for food adulteration detection."  
   *Journal:* *Food Chemistry*, 221, 1236–1244.  
   *Contribution:* This study investigates near-infrared (NIR) and mid-infrared (MIR) spectroscopy coupled with supervised learning algorithms (Linear Discriminant Analysis, Support Vector Machines, and Random Forests) to detect adulteration in milk, honey, and edible oils, demonstrating that machine learning can accurately model physicochemical threshold violations.

4. **Kavitha, M. S., & Sasipraba, T. (2020).**  
   *Title:* "A review on data mining and machine learning techniques for food quality evaluation and classification."  
   *Journal:* *Journal of Food Engineering and Technology*, 9(1), 14–26.  
   *Contribution:* The authors benchmark traditional data mining models (Decision Trees, Naive Bayes, Artificial Neural Networks) on food physical attributes, concluding that tree ensembles provide the optimal compromise between accuracy and interpretability for regulatory personnel.

5. **Lohumi, S., Lee, S., Lee, H., & Cho, B. K. (2015).**  
   *Title:* "A review of hyperspectral imaging for food quality and safety analysis."  
   *Journal:* *Journal of Food Engineering*, 151, 90–103.  
   *Contribution:* Highlights the fusion of spectral and spatial image data with multivariate clustering and classification algorithms to detect adulteration in powdered foods and grains, directly informing our future work recommendations.

---

## 3. Existing System vs. Proposed System

### 3.1 Drawbacks of the Existing System
1. **Centralized Testing Bottlenecks:** Relies on physical chemical testing (Kjeldahl digestion, Gerber butyrometer, HPLC) requiring days of turnaround.
2. **Binary Pass/Fail Rigidity:** Static threshold testing lacks probabilistic nuance and cannot stratify borderline, near-expiry, or temperature-abused consignments.
3. **Lack of Audit Provenance:** Field inspections are frequently recorded on paper manifests or disconnected spreadsheets, leading to chain-of-custody loss.
4. **No Explainability:** Conventional lab assays do not provide algorithmic risk attribution to aid field officers in triage.

### 3.2 Key Innovations of the Proposed System
1. **Multi-Paradigm Intelligence:** Combines supervised classification, unsupervised K-Means clustering, and Apriori association mining.
2. **Tri-Tier Risk Stratification:** Translates calibrated posterior probabilities into Low Risk ($< 0.35$), Medium Risk ($0.35 - 0.65$), and High Risk ($> 0.65$).
3. **Leak-Free ML Pipeline:** Enforces mathematical separation of train/test folds, ensuring SMOTE oversampling is applied strictly to training folds.
4. **Safety-First Optimization:** Formally optimizes for **Recall** on the adulterated class to eliminate dangerous False Negatives.
5. **Real-Time Audit Persistence:** Incorporates SQLite database logging (`predictions.db`) for every single or batch inference.

| Feature / Dimension | Existing Inspection Workflow | Proposed Data Mining System |
|---|---|---|
| **Analytical Latency** | 7 – 21 days (central laboratory) | Real-time (< 50 milliseconds) |
| **Operational Cost** | High (expensive chemical reagents) | Zero per-sample reagent cost |
| **Risk Stratification** | Binary (Pass / Fail) | Tri-Tier (Low / Medium / High Risk) |
| **Inspection Scalability** | Low (bounded by lab capacity) | High (instantaneous batch CSV screening) |
| **Interpretability** | Complex chemical spectra | Interactive feature attribution & XAI |
| **Data Provenance** | Disconnected paper/Excel logs | Automated SQLite database audit log |

---

## 4. System Architecture

The complete system adheres to an 8-stage data mining lifecycle:

```text
[Stage 1: Grounded Data Generation (6000 Samples across 7 FSSAI Categories)]
                             │
                             ▼
[Stage 2: Robust Preprocessing Pipeline (Imputation, IQR Outliers, Scaling, Leak-free SMOTE)]
                             │
                             ▼
[Stage 3: Exploratory Data Analysis & Tri-Method Feature Selection]
                             │
                             ▼
[Stage 4: Multi-Paradigm Mining (7 Classifiers + K-Means Clustering + Apriori Rules)]
                             │
                             ▼
[Stage 5: Safety-First Model Evaluation & Recall-Driven Selection]
                             │
                             ▼
[Stage 6: Prediction Engine with Calibrated Risk Band Stratification]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
[Stage 7: Streamlit Cloud Dashboard]   [Stage 8: SQLite Database Audit Persistence]
```

---

## 5. Module Description

1. **`src/data_generation.py` (Stage 1):** Synthesizes 6,000 observations across milk, oil, spices, grains, honey, beverages, and processed foods with authentic adulterant distributions (starch, detergent, synthetic dyes, argemone oil, invert sugars).
2. **`src/preprocessing.py` (Stage 2):** Implements `ColumnTransformer` with `SimpleImputer`, `IQRClipper`, `StandardScaler`, `OneHotEncoder`, and train-only SMOTE oversampling.
3. **`src/feature_selection.py` (Stage 3):** Generates 8 publication-grade visualization figures and computes a comparative ranking table across Pearson correlation, Chi-Square, and Random Forest Gini importance.
4. **`src/models.py` (Stage 4):** Executes 5-fold `GridSearchCV` on Decision Tree, Random Forest, Naive Bayes, SVM, Logistic Regression, KNN, and Gradient Boosting; evaluates K-Means ($k \in [2, 6]$); mines Apriori association rules.
5. **`src/evaluation.py` (Stage 5):** Evaluates models on hold-out test data; plots overlaid ROC curves and confusion matrices; selects the winning model based on Recall; serializes `best_model_pipeline.joblib`.
6. **`src/predict.py` (Stage 6):** Exposes `predict_single()` and `predict_batch()` returning binary status, continuous risk probability, risk band, and feature attribution drivers.
7. **`app/app.py` (Stage 7):** Interactive 6-tab Streamlit dashboard for single prediction, batch scanning, EDA inspection, model benchmarking, database reports, and project about page.
8. **`src/database.py` (Stage 8):** Manages SQLite `db/predictions.db` audit persistence, daily summaries, category analytics, and full audit export.

---

## 6. Implementation & Algorithmic Details

### 6.1 Input Parameters & Operational Ranges
The system processes 10 fundamental quality indicators:
1. `moisture_content` (% w/w, continuous): Standard limits defined per commodity (e.g. Milk 87–89%, Honey $\le 20\%$, Oil $\le 0.2\%$).
2. `ph_level` (continuous, 2.0 to 10.0): Objective measure of acidity/alkalinity.
3. `storage_temperature` (°C, continuous): Measures cold chain compliance (Milk $\le 6^\circ\text{C}$).
4. `expiry_days_remaining` (integer): Shelf-life tracking (negative values denote expired stock).
5. `preservatives_used` (nominal categorical): `none`, `permitted`, `non-permitted`.
6. `food_category` (nominal categorical): 7 commodities.
7. `color_score` (ordinal Likert 1–5): Organoleptic visual assessment.
8. `smell_score` (ordinal Likert 1–5): Olfactory assessment of freshness vs rancidity.
9. `taste_score` (ordinal Likert 1–5): Palatability and chemical off-flavor screening.
10. `packaging_quality_score` (ordinal Likert 1–5): Seal integrity and tamper evidence.

### 6.2 Preprocessing & Data Leakage Prevention
Let $D = \{(x_i, y_i)\}_{i=1}^N$ be the raw dataset. We partition $D$ using stratified sampling into training partition $D_{train}$ (80%, 4,800 samples) and hold-out test partition $D_{test}$ (20%, 1,200 samples).  
Outlier clipping thresholds are computed strictly on $D_{train}$:
$$\text{Lower}_j = Q_{1, j} - 1.5 \times (Q_{3, j} - Q_{1, j}), \quad \text{Upper}_j = Q_{3, j} + 1.5 \times (Q_{3, j} - Q_{1, j})$$
SMOTE is executed exclusively on $D_{train}$, expanding the minority adulterated class from 1,614 samples to 3,186 samples to match the majority safe class, creating a balanced 6,372-sample training set. The test partition $D_{test}$ remains un-resampled with its original 33.58% positive incidence.

---

## 7. Results & Discussion

### 7.1 Multi-Model Benchmarking on Hold-Out Test Set
All models were evaluated on the independent 1,200-sample test set (797 Safe, 403 Adulterated):

| Model Name | Accuracy | Precision | Recall (Adulterated) | F1-Score | ROC-AUC | 5-Fold CV Recall |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **99.67%** | **99.02%** | **100.00%** | **0.9951** | **1.0000** | **0.9997 ± 0.0006** |
| **Random Forest** | 99.67% | 99.02% | 100.00% | 0.9951 | 0.9999 | 1.0000 ± 0.0000 |
| **Decision Tree** | 99.67% | 99.02% | 100.00% | 0.9951 | 0.9998 | 0.9975 ± 0.0032 |
| **Support Vector Machine (RBF)** | 99.67% | 99.02% | 100.00% | 0.9951 | 0.9985 | 1.0000 ± 0.0000 |
| **K-Nearest Neighbors ($k=7$)** | 99.67% | 99.02% | 100.00% | 0.9951 | 0.9980 | 0.9997 ± 0.0006 |
| **Logistic Regression** | 99.75% | 99.50% | 99.75% | 0.9963 | 0.9999 | 1.0000 ± 0.0000 |
| **Gaussian Naive Bayes** | 99.67% | 99.26% | 99.75% | 0.9950 | 0.9993 | 1.0000 ± 0.0000 |

### 7.2 Safety-First Selection Justification
Gradient Boosting achieved **100.0% Recall** on the adulterated class ($FN = 0$ out of 403 cases) and an **ROC-AUC of 1.0000**. While Logistic Regression achieved marginally higher raw accuracy (99.75% vs 99.67%), it produced a False Negative (1 adulterated sample cleared as safe). In public health enforcement, a single false negative can trigger a fatal food poisoning outbreak. Consequently, Gradient Boosting was selected and serialized as our production engine.

### 7.3 Unsupervised K-Means Clustering Validation
K-Means clustering evaluated across $k \in [2, 6]$ revealed an optimal inflection point at $k=2$ with a maximum Silhouette Score of **0.4131**. The contingency matrix against ground-truth classes demonstrated:
- Cluster 0: 3,168 Safe vs 7 Adulterated (99.78% purity)
- Cluster 1: 3,179 Adulterated vs 18 Safe (99.44% purity)  
This confirms that adulterated samples possess distinct multivariate geometric separation that naturally segregates without supervision.

### 7.4 Apriori Association Rules Discovered
The top mined association rules confirmed key legal violations:
1. `NonPermitted_Preservative => Adulterated` (Support = 15.92%, Confidence = 100.0%, Lift = 2.97)
2. `NonPermitted_Preservative & Expired_Stock => Adulterated` (Support = 9.32%, Confidence = 100.0%, Lift = 2.97)
3. `Expired_Stock & Moisture_Anomalous => Adulterated` (Support = 10.70%, Confidence = 100.0%, Lift = 2.97)
4. `Severe_Sensory_Defect & pH_Anomalous => Adulterated` (Support = 11.85%, Confidence = 99.86%, Lift = 2.97)

---

## 8. Conclusion

This project successfully demonstrates the design, empirical validation, and deployment of a multi-paradigm data mining system for food adulteration risk prediction. By mathematically grounding 6,000 synthetic observations in published FSSAI standards, we constructed a leak-free preprocessing pipeline and trained seven high-performing classification algorithms alongside unsupervised K-Means clustering and Apriori association mining. By enforcing a **Recall-driven model selection protocol (100.0% test recall, 0 False Negatives)**, the system prioritizes consumer protection and public health integrity. The operational workflow is fully realized through a 6-tab Streamlit dashboard backed by persistent SQLite audit logging, providing a practical, scalable decision-support tool for food safety officers.

---

## 9. Future Enhancements

While the current framework provides rapid field screening, it possesses methodological reliance on human sensory scoring. To transition from a decision-support prototype to an industrial cyber-physical inspection platform, future enhancements include:
1. **Handheld Near-Infrared (NIR) Spectroscopy:** Interfacing low-cost MEMS-based NIR spectrometers (900–1700 nm) to capture optical absorption spectra of O-H and C-H bonds, digitizing moisture, fat, and sugar content directly and eliminating sensory subjectivity.
2. **Electronic Nose (E-Nose) Sensor Arrays:** Deploying metal-oxide-semiconductor (MOS) gas sensor arrays to detect volatile organic compounds (VOCs) associated with lipid rancidity and microbial souring.
3. **IoT Cold-Chain Telemetry:** Integrating RFID temperature data-loggers into food shipping crates to feed continuous thermal histories directly into our risk engine.
4. **Cloud Database Migration:** Transitioning local SQLite storage to enterprise managed PostgreSQL or Supabase instances to enable statewide surveillance synchronization.

---

## 10. References (IEEE Format)

1. [1] D. I. Ellis, V. L. Brewster, W. B. Dunn, J. W. Allwood, and R. Goodacre, "Point-and-shoot: rapid qualitative and quantitative detection of food adulteration using spectroscopic and sensor technologies," *Chemical Society Reviews*, vol. 41, no. 2, pp. 570–585, 2012.
2. [2] M. P. Callao and I. Ruisánchez, "An overview of multivariate qualitative methods for food fraud detection," *Food Control*, vol. 86, pp. 318–324, 2018.
3. [3] C. F. Nascimento, P. M. Santos, E. R. Pereira-Filho, and F. R. Rocha, "Recent advances in infrared spectroscopy and chemometrics for food adulteration detection," *Food Chemistry*, vol. 221, pp. 1236–1244, 2017.
4. [4] M. S. Kavitha and T. Sasipraba, "A review on data mining and machine learning techniques for food quality evaluation and classification," *Journal of Food Engineering and Technology*, vol. 9, no. 1, pp. 14–26, 2020.
5. [5] S. Lohumi, S. Lee, H. Lee, and B. K. Cho, "A review of hyperspectral imaging for food quality and safety analysis," *Journal of Food Engineering*, vol. 151, pp. 90–103, 2015.
6. [6] Food Safety and Standards Authority of India (FSSAI), *Food Safety and Standards Act, 2006*, Ministry of Health and Family Welfare, Government of India, New Delhi, 2006.
7. [7] Food Safety and Standards Authority of India (FSSAI), *National Milk Safety Survey (NMSS) 2019 Executive Summary*, FSSAI, New Delhi, India, 2019.
8. [8] Bureau of Indian Standards (BIS), *IS 6273: Guide for Sensory Evaluation of Foods*, Part 1 and Part 2, Manak Bhavan, New Delhi, India.

---

## 11. Appendix: System Codebase Index

- `src/data_generation.py`: Statistical grounded synthesis of 6,000 records.
- `src/preprocessing.py`: Scikit-Learn `ColumnTransformer`, `IQRClipper`, and leak-free SMOTE.
- `src/feature_selection.py`: 8 EDA figures and tri-method selection comparison.
- `src/models.py`: 5-fold GridSearchCV classification, K-Means clustering, and Apriori mining.
- `src/evaluation.py`: Model evaluation, overlaid ROC curves, confusion matrix grid, winning model serialization.
- `src/predict.py`: Single and batch inference engine with 3-tier risk banding.
- `src/database.py`: SQLite `db/predictions.db` audit logging and aggregation queries.
- `app/app.py`: Streamlit 6-tab interactive web dashboard.
- `notebooks/01_data_generation_and_dictionary.ipynb`
- `notebooks/02_preprocessing_and_pipeline.ipynb`
- `notebooks/03_eda_and_feature_selection.ipynb`
- `notebooks/04_model_building_and_mining.ipynb`
- `notebooks/05_evaluation_and_explainability.ipynb`
