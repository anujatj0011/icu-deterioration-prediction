# Explainable AI for Early ICU Deterioration Prediction

End-to-end machine learning pipeline for predicting ICU deterioration from the **first 6 hours of an ICU admission**, using MIMIC-IV v3.1. The project focuses on a problem that matters in real predictive systems: making useful predictions early while preventing temporal leakage and keeping model outputs interpretable.

**46,982 ICU stays · ~210 engineered features · XGBoost · ROC-AUC 0.825 · PR-AUC 0.716 · Recall 73.6% · SHAP · probability calibration**

> **Research prototype only.** This system was developed on retrospective MIMIC-IV data and has not been clinically validated or approved for patient-care decisions.

### Explore the project

**[Open the research workspace](https://anujatj0011-icu-deterioration-prediction-app-01bzpt.streamlit.app/)** · **[Technical guide](docs/TECHNICAL_GUIDE.md)** · **[Key findings](docs/KEY_FINDINGS.md)** · **[Execution guide](docs/EXECUTION_GUIDE.md)**

The Streamlit interface separates public demonstration from restricted local research. Public mode displays an empty workspace without fabricated patients or scores.

## Application workspace

**ICU Observatory** provides a responsive patient workspace, calibrated-score visualization, retrospective cohort filters, underlying-model SHAP contributions and research-use limitations. The implementation stays in `app.py`; preprocessing, training and evaluation scripts remain unchanged.

### Public deployment

Install `requirements.txt` and run `python -m streamlit run app.py`. In Streamlit Community Cloud, keep `app.py` as the entry point and leave `ICU_ENABLE_LOCAL_RESEARCH` unset. Push these source changes and reboot the existing deployment if needed. Do not publish `data/`, credentials or model bundles. No extra application modules are required.

### Local research setup

Generate the original pipeline artifacts, including `python src/08_prepare_app.py`. The app expects these trusted files in `data/` beside `app.py`:

- `app_patient_monitor.csv`
- `app_test_features_imputed.parquet`
- `app_test_features_raw.parquet`
- `app_underlying_model.joblib`

Explicitly enable local research before launching:

```powershell
# Windows PowerShell
$env:ICU_ENABLE_LOCAL_RESEARCH="1"
python -m streamlit run app.py
```

```bash
# macOS / Linux
ICU_ENABLE_LOCAL_RESEARCH=1 python -m streamlit run app.py
```

Existing records use their original stored calibrated probabilities. Advanced CSV inference additionally requires the original `data/model_artifacts.joblib` with `calibrated_model`. Upload exactly one numeric, finite, already imputed feature row with the complete original schema in order. The interface provides a download of feature names without patient values. It invokes the original calibrated estimator and uses class `1` as the deterioration outcome.

Raw-vitals inference is unavailable because the original bundle does not export the fitted imputer. Do not invent missing engineered inputs or fit replacement imputation on test data. SHAP contributions explain the underlying estimator before calibration, not percentage-point changes in the calibrated score or medical causality.

Use a trusted model-compatible training environment for saved estimators. Load only trusted joblib files. Local mode has no authentication layer; use an authorized single-tenant environment because Streamlit caches are shared within a server process. It must not be enabled on the public deployment.

**Validation boundary:** public startup and input-validation smoke checks passed. Original private-model inference and browser-level visual rendering have not been independently verified. No clinical validation is claimed.

## What this project demonstrates

- **Temporal prediction design:** features are restricted to the first 6 ICU hours, with deterioration evaluated over 24-, 36-, and 48-hour horizons.
- **Leakage controls:** dedicated feature- and label-level audits check temporal integrity before modelling.
- **Model comparison:** Logistic Regression, Random Forest and XGBoost are evaluated rather than presenting a single model in isolation.
- **Imbalanced classification:** performance is evaluated with recall, F1, ROC-AUC and PR-AUC alongside accuracy.
- **Probability calibration:** isotonic calibration is used so predicted risk is more meaningful than an uncalibrated class score.
- **Explainability:** global and patient-level SHAP analyses show which variables drive predictions.
- **Prototype delivery:** a Streamlit risk monitor turns model outputs into an interpretable workflow rather than stopping at a notebook.

## 24-hour results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.742 | 0.603 | 0.731 | 0.661 | 0.812 | 0.692 |
| Random Forest | **0.755** | **0.628** | 0.701 | 0.663 | 0.818 | 0.702 |
| **XGBoost** | 0.751 | 0.615 | **0.736** | **0.670** | **0.825** | **0.716** |

XGBoost was selected as the primary model because it provided the strongest overall discrimination and PR-AUC while retaining the highest recall of the three models. In this use case, missed deterioration events are particularly important, so accuracy alone is not an adequate selection criterion.

![ROC curves comparing the three models](results/roc_curves_full.png)

### Prediction-horizon sensitivity

| Horizon | ICU stays | Positive rate | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| 24 h | 46,982 | 34.4% | 0.736 | 0.670 | 0.825 | 0.716 |
| 36 h | 40,099 | 42.3% | 0.727 | 0.707 | 0.819 | 0.764 |
| 48 h | 33,891 | 51.9% | 0.725 | 0.738 | 0.813 | 0.815 |

![XGBoost horizon sensitivity](results/horizon_sensitivity_roc.png)

## System design

```text
MIMIC-IV v3.1
      │
      ▼
Cohort construction
      │
      ▼
0–6 h vitals, labs & demographics
      │
      ▼
Preprocessing & feature engineering
      │
      ├──► Feature leakage audit
      │
      ▼
24 / 36 / 48 h deterioration labels
      │
      ├──► Label leakage audit
      │
      ▼
LR / Random Forest / XGBoost
      │
      ▼
Evaluation + probability calibration
      │
      ▼
Global & patient-level SHAP
      │
      ▼
Streamlit risk-monitor prototype
```

A deterioration event is defined from mortality, vasopressor requirement and mechanical ventilation. The design intentionally separates the **observation window** from the **prediction horizon** so future information cannot become a predictor.

## Explainability

SHAP analysis is used both globally and at individual-patient level. Important model drivers include lactate, neurological status (GCS), systolic blood pressure and respiratory rate.

![Global SHAP feature importance](results/shap_feature_importance_full.png)

![SHAP summary](results/shap_summary_full.png)

The Streamlit prototype also generates patient-level explanations and separates features that increase versus decrease predicted risk. SHAP values are treated as explanations of the model's prediction, **not as causal or medical explanations**.

## Calibration

For a risk model, ranking patients correctly is not enough: a predicted probability should also have a meaningful relationship with observed risk. The pipeline therefore evaluates probability calibration and applies isotonic calibration.

![Calibration curves before and after calibration](results/calibration_curves_full_before_after.png)

## Project context & my contribution

This project originated as a two-person MSc Data & Computational Science project at **University College Dublin**.

**My technical contribution (Anuja Thuraiyur Jayakumar):** I led and implemented the end-to-end technical work: data extraction, cohort construction, preprocessing, feature and label engineering, temporal leakage auditing, model development and comparison, evaluation, probability calibration, SHAP explainability, sensitivity analysis, and the Streamlit risk-monitoring prototype.

**Project partner contribution (Ruthvik Gowda Bageri Manjunath):** project poster and README/documentation contributions.

The original collaborative academic repository is preserved through this fork's GitHub history and upstream relationship.

## Repository structure

```text
.
├── app.py
├── run_pipeline.py
├── requirements.txt
├── src/
│   ├── 01_data_extraction.py
│   ├── 02_preprocessing.py
│   ├── 03_leakage_audit.py
│   ├── 04_prediction_horizons.py
│   ├── 05_horizon_sensitivity.py
│   ├── 06_model_evaluation.py
│   ├── 07_patient_explanations.py
│   └── 08_prepare_app.py
├── results/
├── docs/
├── Literature_Review/
├── poster/
└── assets/
```

## Reproducing the pipeline

### Prerequisites

- Python 3.8+
- access to **MIMIC-IV v3.1** through PhysioNet
- Google Cloud / BigQuery access as required by the extraction stage
- sufficient local memory for preprocessing and model development

Install dependencies:

```bash
git clone https://github.com/anujatj0011/icu-deterioration-prediction.git
cd icu-deterioration-prediction

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Configure the required GCP credentials/project settings, obtain authorised MIMIC-IV access, then run the pipeline from the repository root:

```bash
python run_pipeline.py
```

After generating app artifacts, enable local research as described above, then run:

```bash
streamlit run app.py
```

For stage-by-stage instructions, see [Execution Guide](docs/EXECUTION_GUIDE.md). For implementation details, see [Technical Guide](docs/TECHNICAL_GUIDE.md), and for interpretation of the outputs see [Key Findings](docs/KEY_FINDINGS.md).

## Data access and privacy

**No MIMIC-IV patient-level data is included in this repository.** MIMIC-IV is a credentialed PhysioNet dataset. Anyone reproducing the work must obtain access directly from PhysioNet and comply with the applicable data-use requirements.

This repository contains source code, documentation and aggregate model results/visualisations only. Do not commit raw or derived patient-level MIMIC data to this repository.

## Limitations

- Results are retrospective and come from MIMIC-IV rather than prospective clinical deployment.
- The model has not undergone external validation on an independent hospital population.
- The outcome combines multiple deterioration events and therefore should not be interpreted as a diagnosis.
- SHAP explains model behaviour; it does not establish causality.
- Alert thresholds in the Streamlit application are prototype controls and are not clinically validated.
- Performance may change under dataset shift, different clinical workflows or different patient populations.

## Documentation

- [Key findings](docs/KEY_FINDINGS.md)
- [Technical guide](docs/TECHNICAL_GUIDE.md)
- [Execution guide](docs/EXECUTION_GUIDE.md)
- [Model comparison data](results/model_comparison_full.csv)
- [Prediction-horizon results](results/horizon_sensitivity_xgboost.csv)

## Tech stack

**Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · BigQuery · Streamlit · Matplotlib · joblib**

---

**Dataset:** MIMIC-IV v3.1, PhysioNet  
**Status:** Research implementation complete  
**Clinical use:** Not validated for clinical deployment
