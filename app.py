from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.pipeline import Pipeline

from demo_data import demo_frame

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
.hero p {font-size: 1.02rem; opacity: .82; margin-bottom: .25rem;}\n[data-testid="stMetric"] {box-shadow: 0 2px 10px rgba(0,0,0,.03);}\n[data-testid="stSidebar"] {border-right: 1px solid rgba(127,127,127,.12);}\n[data-baseweb="tab-list"] {gap: .65rem;}
</style>
""", unsafe_allow_html=True)


def humanize_feature(feature: str):
    units = {"heart_rate":"bpm","sbp":"mmHg","dbp":"mmHg","resp_rate":"breaths/min","spo2":"%",
             "temperature_f":"°F","lactate":"mmol/L","wbc":"10³/µL","creatinine":"mg/dL",
             "platelets":"10³/µL","bicarbonate":"mmol/L","potassium":"mmol/L","sodium":"mmol/L",
             "age":"years","shock_index":""}
    base = feature
    for suffix in ["_mean","_min","_max","_last","_count","_missing"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    unit = "" if feature.endswith(("_count","_missing")) else units.get(base,"")
    label = feature
    for suffix,repl in {"_mean":" mean","_min":" minimum","_max":" maximum","_last":" latest",
                        "_count":" measurement count","_missing":" missingness indicator"}.items():
        if label.endswith(suffix):
            label=label[:-len(suffix)]+repl; break
    label=label.replace("first_careunit_","Care unit: ").replace("admission_type_","Admission type: ")
    return label.replace("_"," ").strip().capitalize(), unit


def format_value(feature, value):
    label, unit = humanize_feature(feature)
    if pd.isna(value): return label, "Missing"
    value=float(value)
    rendered=f"{value:.0f}" if abs(value)>=100 else (f"{value:.1f}" if abs(value)>=10 else f"{value:.2f}")
    return label, rendered + (f" {unit}" if unit else "")


def risk_category(probability, moderate, high):
    return "High" if probability >= high else ("Moderate" if probability >= moderate else "Low")


def local_artifacts_available():
    return all((DATA_DIR / f).exists() for f in [
        "app_patient_monitor.csv","app_test_features_imputed.parquet",
        "app_test_features_raw.parquet","app_underlying_model.joblib"])


@st.cache_resource
def load_model():
    return joblib.load(DATA_DIR / "app_underlying_model.joblib")


@st.cache_data
def load_local_data():
    return (pd.read_csv(DATA_DIR/"app_patient_monitor.csv"),
            pd.read_parquet(DATA_DIR/"app_test_features_imputed.parquet"),
            pd.read_parquet(DATA_DIR/"app_test_features_raw.parquet"))


def shap_for_patient(model, x_row):
    if isinstance(model, Pipeline):
        clf=model.named_steps["clf"]; scaler=model.named_steps["scale"]
        transformed=pd.DataFrame(scaler.transform(x_row),columns=x_row.columns,index=x_row.index)
        explanation=shap.LinearExplainer(clf,transformed)(transformed)[0]; plot_data=transformed.iloc[0].to_numpy()
    else:
        explanation=shap.TreeExplainer(model)(x_row)[0]; plot_data=x_row.iloc[0].to_numpy()
    if np.asarray(explanation.values).ndim>1: explanation=explanation[...,1]
    return shap.Explanation(values=np.asarray(explanation.values),base_values=explanation.base_values,
                            data=plot_data,feature_names=list(x_row.columns))


def local_contributions(explanation, raw_row, top_n=10):
    vals=np.asarray(explanation.values).reshape(-1); names=list(explanation.feature_names)
    rows=[]
    for rank,j in enumerate(np.argsort(np.abs(vals))[::-1][:top_n],1):
        feature=names[j]; label,pv=format_value(feature,raw_row.get(feature,np.nan)); v=float(vals[j])
        rows.append({"Rank":rank,"Feature":label,"Patient value":pv,"SHAP value":v,
                     "Direction":"Increases risk" if v>0 else "Decreases risk"})
    return pd.DataFrame(rows)


# Public deployments have no restricted artifacts, so they automatically enter synthetic mode.
demo_mode = not local_artifacts_available()
if demo_mode:
    monitor=demo_frame()
    x_imp=x_raw=underlying_model=None
else:
    monitor,x_imp,x_raw=load_local_data(); underlying_model=load_model()

st.markdown("""
<div class="hero"><div class="eyebrow">Explainable machine learning · research prototype</div>
<h1>ICU Early Warning System</h1>
<p>Estimate deterioration risk from information available during the first 6 hours of ICU admission, then inspect the factors driving each prediction.</p></div>
""", unsafe_allow_html=True)

m1,m2,m3,m4=st.columns(4)
m1.metric("Observation window","First 6 h"); m2.metric("Prediction horizon","Next 24 h")
m3.metric("Research ROC-AUC","0.825"); m4.metric("Research recall","73.6%")

if demo_mode:
    st.warning("PUBLIC DEMO · SYNTHETIC DATA — Synthetic profiles and illustrative scores only. No MIMIC-IV patient data, MIMIC-derived patient artifacts, or trained MIMIC model is included.")
else:
    st.info("LOCAL RESEARCH MODE — Restricted MIMIC-derived artifacts are loaded only from this local environment. Do not deploy these artifacts publicly.")

with st.sidebar:
    st.markdown("### ICU Early Warning"); st.caption("XGBoost research pipeline · calibrated risk · SHAP")
    st.divider(); st.header("Alert settings")
    moderate_threshold=st.slider("Moderate-risk threshold",0.05,0.70,0.30,0.05)
    high_threshold=st.slider("High-risk alert threshold",0.10,0.95,0.50,0.05)
    if moderate_threshold>=high_threshold: st.error("Moderate threshold must be lower than high-risk threshold."); st.stop()
    st.caption("Prototype thresholds — not clinically validated.")

monitor=monitor.copy()
monitor["risk_category"]=monitor["calibrated_probability"].map(lambda p:risk_category(float(p),moderate_threshold,high_threshold))
monitor_tab,patient_tab,model_tab=st.tabs(["Population Risk Monitor","Patient Explanation","Model Card"])

with monitor_tab:
    n_high=int((monitor.calibrated_probability>=high_threshold).sum())
    n_mod=int(((monitor.calibrated_probability>=moderate_threshold)&(monitor.calibrated_probability<high_threshold)).sum())
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Profiles monitored",f"{len(monitor):,}"); c2.metric("High-risk alerts",f"{n_high:,}")
    c3.metric("Moderate risk",f"{n_mod:,}"); c4.metric("Low risk",f"{len(monitor)-n_high-n_mod:,}")
    display=monitor[["display_id","calibrated_probability","risk_category"]].copy()
    display.calibrated_probability=display.calibrated_probability.map(lambda x:f"{x:.1%}")
    display.columns=["Profile","Illustrative risk score","Risk category"]
    st.subheader("Priority review queue")
    st.caption("In public demo mode these are synthetic interface examples, not real patients or model predictions.")
    st.dataframe(display,use_container_width=True,hide_index=True,height=360)

with patient_tab:
    selected_id=st.selectbox("Select profile",monitor.display_id.tolist())
    mrow=monitor.loc[monitor.display_id==selected_id].iloc[0]
    probability=float(mrow.calibrated_probability); category=risk_category(probability,moderate_threshold,high_threshold)
    a,b,c=st.columns(3); a.metric("Illustrative risk score" if demo_mode else "Predicted 24h risk",f"{probability:.1%}"); b.metric("Risk category",category); c.metric("Prediction horizon","Next 24 h")
    if demo_mode:
        st.subheader("Illustrative explanation")
        st.caption("The factors below are authored synthetic examples for demonstrating the intended explanation workflow; they are not SHAP outputs.")
        left,right=st.columns(2)
        with left:
            st.markdown("**Factors increasing illustrative score**")
            for item in mrow["drivers_up"]: st.write(f"↑ {item}")
        with right:
            st.markdown("**Factors decreasing illustrative score**")
            for item in mrow["drivers_down"]: st.write(f"↓ {item}")
        st.subheader("Synthetic profile values")
        feature_cols=["age","heart_rate_last","sbp_last","resp_rate_last","spo2_last","lactate_last","creatinine_last","platelets_last","gcs_verbal_last"]
        rows=[]
        for f in feature_cols:
            label,value=format_value(f,mrow[f]); rows.append({"Feature":label,"Synthetic value":value})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    else:
        x_row=x_imp.loc[x_imp.display_id==selected_id].drop(columns=["display_id"])
        raw_row=x_raw.loc[x_raw.display_id==selected_id].drop(columns=["display_id"]).iloc[0]
        with st.spinner("Generating patient-level SHAP explanation..."):
            explanation=shap_for_patient(underlying_model,x_row); contrib=local_contributions(explanation,raw_row,12)
        st.subheader("Why was this patient assigned this risk?")
        st.caption("SHAP explains model behaviour, not medical causality.")
        pos=contrib[contrib["SHAP value"]>0].head(5); neg=contrib[contrib["SHAP value"]<0].head(5)
        left,right=st.columns(2)
        with left:
            st.markdown("**Factors increasing predicted risk**")
            for _,r in pos.iterrows(): st.write(f"↑ **{r['Feature']}** — {r['Patient value']}")
        with right:
            st.markdown("**Factors decreasing predicted risk**")
            for _,r in neg.iterrows(): st.write(f"↓ **{r['Feature']}** — {r['Patient value']}")
        readable=[humanize_feature(f)[0] for f in explanation.feature_names]
        exp=shap.Explanation(values=explanation.values,base_values=explanation.base_values,data=explanation.data,feature_names=readable)
        shap.plots.waterfall(exp,max_display=10,show=False); plt.tight_layout(); st.pyplot(plt.gcf(),clear_figure=True,use_container_width=True)

with model_tab:
    st.subheader("Model card")
    st.markdown("""
**Research question** — Can information available in the first 6 ICU hours identify elevated deterioration risk over the following 24 hours?

**Dataset & cohort** — MIMIC-IV v3.1; first ICU stay per patient with ≥12 hours ICU length of stay. The 24-hour analysis contains **46,982 eligible ICU stays**.\n\n**Temporal design** — Time-varying predictors are restricted to hours **0–6** after ICU admission. The primary outcome window is the subsequent **24 hours (6–30h)**.\n\n**Outcome** — Composite deterioration event defined from mortality, vasopressor requirement, or mechanical ventilation within the prediction window.\n\n**Research pipeline** — Logistic Regression, Random Forest and XGBoost comparison; explicit temporal leakage audits; isotonic probability calibration; global and patient-level SHAP.

**Held-out research results** — ROC-AUC **0.825** · PR-AUC **0.716** · Recall **73.6%** · F1 **0.670**.

**Public demo boundary** — The deployed interface uses hand-authored synthetic profiles and illustrative risk scores. The restricted MIMIC-derived model and patient-level artifacts are not packaged with the public application.

**Limitations** — Retrospective MIMIC-IV analysis; no prospective or external validation. The research model is not a medical device and is not intended for patient-care decisions.
""")

st.divider()
st.caption("Portfolio research prototype · MIMIC-IV v3.1 research results · public interface uses synthetic demo profiles only · not for clinical use")
