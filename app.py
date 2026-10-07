from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.pipeline import Pipeline

DATA_DIR = Path("./data")

st.set_page_config(
    page_title="ICU Early Warning | Explainable ML",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetric"] {background: rgba(127,127,127,.06); border: 1px solid rgba(127,127,127,.18); padding: 14px 16px; border-radius: 12px;}
.hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(127,127,127,.20); border-radius: 16px; margin-bottom: 1rem;}
.eyebrow {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7; font-weight: 700;}
.hero h1 {margin: .25rem 0 .45rem 0;}
.hero p {font-size: 1.02rem; opacity: .82; margin-bottom: .25rem;}
</style>
""", unsafe_allow_html=True)


def humanize_feature(feature: str):
    units = {
        "heart_rate": "bpm",
        "sbp": "mmHg",
        "dbp": "mmHg",
        "resp_rate": "breaths/min",
        "spo2": "%",
        "temperature_f": "°F",
        "gcs_eye": "",
        "gcs_verbal": "",
        "gcs_motor": "",
        "lactate": "mmol/L",
        "wbc": "10³/µL",
        "creatinine": "mg/dL",
        "platelets": "10³/µL",
        "bicarbonate": "mmol/L",
        "potassium": "mmol/L",
        "sodium": "mmol/L",
        "age": "years",
        "shock_index": "",
    }
    base = feature
    for suffix in ["_mean", "_min", "_max", "_last", "_count", "_missing"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    unit = "" if feature.endswith(("_count", "_missing")) else units.get(base, "")

    label = feature
    replacements = {
        "_mean": " mean",
        "_min": " minimum",
        "_max": " maximum",
        "_last": " latest",
        "_count": " measurement count",
        "_missing": " missingness indicator",
    }
    for suffix, repl in replacements.items():
        if label.endswith(suffix):
            label = label[: -len(suffix)] + repl
            break
    label = label.replace("first_careunit_", "Care unit: ")
    label = label.replace("admission_type_", "Admission type: ")
    label = label.replace("_", " ").strip().capitalize()
    return label, unit


def format_value(feature: str, value):
    label, unit = humanize_feature(feature)
    if pd.isna(value):
        return label, "Missing"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return label, str(value)
    if feature.endswith("_count") or feature.endswith("_missing"):
        rendered = str(int(round(value)))
    elif abs(value) >= 100:
        rendered = f"{value:.0f}"
    elif abs(value) >= 10:
        rendered = f"{value:.1f}"
    else:
        rendered = f"{value:.2f}"
    if unit:
        rendered += f" {unit}"
    return label, rendered


@st.cache_resource
def load_model():
    return joblib.load(DATA_DIR / "app_underlying_model.joblib")


@st.cache_data
def load_data():
    monitor = pd.read_csv(DATA_DIR / "app_patient_monitor.csv")
    x_imp = pd.read_parquet(DATA_DIR / "app_test_features_imputed.parquet")
    x_raw = pd.read_parquet(DATA_DIR / "app_test_features_raw.parquet")
    return monitor, x_imp, x_raw


def ensure_files():
    required = [
        "app_patient_monitor.csv",
        "app_test_features_imputed.parquet",
        "app_test_features_raw.parquet",
        "app_underlying_model.joblib",
    ]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        st.error(
            "Missing local app artifacts: " + ", ".join(missing)
            + ". Run src/08_prepare_app.py first."
        )
        st.stop()


def risk_category(probability: float, moderate: float, high: float):
    if probability >= high:
        return "High"
    if probability >= moderate:
        return "Moderate"
    return "Low"


def shap_for_patient(model, x_row: pd.DataFrame):
    """Return a one-row SHAP Explanation for the underlying tuned model."""
    if isinstance(model, Pipeline):
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scale"]
        transformed = pd.DataFrame(
            scaler.transform(x_row), columns=x_row.columns, index=x_row.index
        )
        explainer = shap.LinearExplainer(clf, transformed)
        explanation = explainer(transformed)[0]
        plot_data = transformed.iloc[0].to_numpy()
    else:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(x_row)[0]
        plot_data = x_row.iloc[0].to_numpy()

    if np.asarray(explanation.values).ndim > 1:
        explanation = explanation[..., 1]

    return shap.Explanation(
        values=np.asarray(explanation.values),
        base_values=explanation.base_values,
        data=plot_data,
        feature_names=list(x_row.columns),
    )


def local_contributions(explanation, raw_row: pd.Series, top_n=10):
    vals = np.asarray(explanation.values).reshape(-1)
    names = list(explanation.feature_names)
    order = np.argsort(np.abs(vals))[::-1][:top_n]
    rows = []
    for rank, j in enumerate(order, start=1):
        feature = names[j]
        label, patient_value = format_value(feature, raw_row.get(feature, np.nan))
        v = float(vals[j])
        rows.append({
            "Rank": rank,
            "Feature": label,
            "Patient value": patient_value,
            "SHAP value": v,
            "Direction": "Increases risk" if v > 0 else "Decreases risk",
        })
    return pd.DataFrame(rows)


ensure_files()
monitor, x_imp, x_raw = load_data()
underlying_model = load_model()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Explainable machine learning · research prototype</div>
  <h1>ICU Early Warning System</h1>
  <p>Estimate deterioration risk from information available during the first 6 hours of ICU admission, then inspect the factors driving each prediction.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Observation window", "First 6 h")
m2.metric("Prediction horizon", "Next 24 h")
m3.metric("Test ROC-AUC", "0.825")
m4.metric("Test recall", "73.6%")

st.info(
    "Portfolio demonstration of a retrospective MIMIC-IV research workflow. "
    "Predictions and thresholds are not clinically validated and must not be used for patient care."
)

with st.sidebar:
    st.markdown("### ICU Early Warning")
    st.caption("XGBoost · calibrated risk · SHAP")
    st.divider()
    st.header("Alert settings")
    moderate_threshold = st.slider("Moderate-risk threshold", 0.05, 0.70, 0.30, 0.05)
    high_threshold = st.slider("High-risk alert threshold", 0.10, 0.95, 0.50, 0.05)
    if moderate_threshold >= high_threshold:
        st.error("Moderate threshold must be lower than the high-risk threshold.")
        st.stop()
    st.caption("Prototype thresholds — not clinically validated.")

# Recalculate categories from the interactive thresholds.
monitor = monitor.copy()
monitor["risk_category"] = monitor["calibrated_probability"].map(
    lambda p: risk_category(float(p), moderate_threshold, high_threshold)
)

monitor_tab, patient_tab, model_tab = st.tabs(["Population Risk Monitor", "Patient Explanation", "Model Card"])

with monitor_tab:
    n_high = int((monitor["calibrated_probability"] >= high_threshold).sum())
    n_moderate = int(((monitor["calibrated_probability"] >= moderate_threshold) &
                      (monitor["calibrated_probability"] < high_threshold)).sum())
    n_low = len(monitor) - n_high - n_moderate

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients monitored", f"{len(monitor):,}")
    c2.metric("High-risk alerts", f"{n_high:,}")
    c3.metric("Moderate risk", f"{n_moderate:,}")
    c4.metric("Low risk", f"{n_low:,}")

    if n_high:
        st.error(
            f"⚠ {n_high:,} patients currently exceed the high-risk threshold "
            f"of {high_threshold:.0%}."
        )
    else:
        st.success("No patients exceed the current high-risk alert threshold.")

    display = monitor[["display_id", "calibrated_probability", "risk_category"]].copy()
    display["calibrated_probability"] = display["calibrated_probability"].map(lambda x: f"{x:.1%}")
    display.columns = ["Patient", "24h predicted risk", "Risk category"]
    st.subheader("Priority review queue")
    st.caption("Patients are ordered by calibrated predicted risk. Display IDs are non-identifying app labels.")
    st.dataframe(display, use_container_width=True, hide_index=True, height=520)

with patient_tab:
    default_id = str(monitor.iloc[0]["display_id"])
    selected_id = st.selectbox(
        "Select patient",
        monitor["display_id"].tolist(),
        index=monitor["display_id"].tolist().index(default_id),
    )

    mrow = monitor.loc[monitor["display_id"] == selected_id].iloc[0]
    probability = float(mrow["calibrated_probability"])
    category = risk_category(probability, moderate_threshold, high_threshold)

    a, b, c = st.columns(3)
    a.metric("24h deterioration risk", f"{probability:.1%}")
    b.metric("Risk category", category)
    c.metric("Prediction horizon", "Next 24 h")

    if category == "High":
        st.error(f"⚠ HIGH-RISK ALERT: {selected_id} exceeds the current alert threshold.")
    elif category == "Moderate":
        st.warning(f"Moderate predicted risk for {selected_id}.")
    else:
        st.success(f"Lower predicted risk for {selected_id}.")

    x_row = x_imp.loc[x_imp["display_id"] == selected_id].drop(columns=["display_id"])
    raw_row = x_raw.loc[x_raw["display_id"] == selected_id].drop(columns=["display_id"]).iloc[0]

    with st.spinner("Generating patient-level SHAP explanation..."):
        explanation = shap_for_patient(underlying_model, x_row)
        contrib = local_contributions(explanation, raw_row, top_n=12)

    st.subheader("Why was this patient assigned this risk?")
    st.caption(
        "SHAP shows which available features pushed the model toward or away from "
        "a deterioration prediction. It does not identify the medical cause of deterioration."
    )

    positive = contrib[contrib["SHAP value"] > 0].head(5)
    negative = contrib[contrib["SHAP value"] < 0].head(5)
    left, right = st.columns(2)
    with left:
        st.markdown("**Factors increasing predicted risk**")
        if positive.empty:
            st.write("No positive contributors among the strongest local features.")
        else:
            for _, r in positive.iterrows():
                st.write(f"↑ **{r['Feature']}** — {r['Patient value']}")
    with right:
        st.markdown("**Factors decreasing predicted risk**")
        if negative.empty:
            st.write("No negative contributors among the strongest local features.")
        else:
            for _, r in negative.iterrows():
                st.write(f"↓ **{r['Feature']}** — {r['Patient value']}")

    st.subheader("Patient-level SHAP waterfall")
    readable_names = [humanize_feature(f)[0] for f in explanation.feature_names]
    readable_explanation = shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=readable_names,
    )
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(readable_explanation, max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)

with model_tab:
    st.subheader("Model card")
    st.markdown("""
**Intended purpose**  
Research prototype for studying whether early ICU measurements can identify patients at elevated risk of deterioration.

**Input window**  
Vitals, laboratory measurements and demographics available during hours 0–6 after ICU admission.

**Outcome**  
Composite deterioration event during the subsequent 24-hour prediction window, including mortality, vasopressor requirement or mechanical ventilation.

**Primary model**  
XGBoost selected after comparison with Logistic Regression and Random Forest. Probabilities are calibrated using isotonic calibration.

**Held-out 24-hour results**  
ROC-AUC **0.825** · PR-AUC **0.716** · Recall **73.6%** · F1 **0.670**

**Safety and limitations**  
Retrospective MIMIC-IV analysis; no prospective or external validation. SHAP explains model behaviour rather than causality. Prototype alert thresholds are not clinical recommendations.
""")

st.divider()
st.caption(
    "Built as an explainable-ML research prototype using MIMIC-IV v3.1 · "
    "Not clinically validated · No patient-level MIMIC data is distributed with this repository."
)
