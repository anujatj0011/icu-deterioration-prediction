"""Research UI; no synthetic records or replacement preprocessing."""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
DATA_DIR = Path(__file__).resolve().parent / 'data'

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
    import shap
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




import numpy as np
import pandas as pd


def validate_input(frame, columns):
    if len(frame) != 1 or list(frame.columns) != columns:
        raise ValueError("Provide exactly one row with the complete original feature schema in order.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("All prepared features must be numeric.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Prepared features must contain no missing or infinite values.")


def validate_artifacts(monitor, imputed, raw):
    if monitor.empty:
        raise ValueError("No research records available.")
    for frame in (monitor, imputed, raw):
        if "display_id" not in frame or frame.display_id.isna().any() or frame.display_id.duplicated().any():
            raise ValueError("Research IDs must be present and unique.")
    if set(monitor.display_id) != set(imputed.display_id) or set(monitor.display_id) != set(raw.display_id):
        raise ValueError("Research record IDs do not align.")
    if list(imputed.columns) != list(raw.columns):
        raise ValueError("Raw and imputed schemas differ.")
    probability = pd.to_numeric(monitor.calibrated_probability, errors="raise")
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError("Calibrated scores must be finite probabilities.")

st.set_page_config(page_title="ICU Observatory · Research", page_icon="◈", layout="wide")
st.markdown('''<style>
.block-container{max-width:1400px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.035em}h1{font-size:3rem!important}
[data-testid="stSidebar"]{border-right:1px solid #DDE6EC}
[data-testid="stMetric"]{background:white;border:1px solid #DDE6EC;border-radius:16px;padding:20px}
.eyebrow{color:#007F78;font-size:.73rem;letter-spacing:.16em;font-weight:750}
.hero{background:linear-gradient(120deg,#FFF 55%,#E4F1F1);border:1px solid #DDE6EC;border-radius:22px;padding:32px;margin:16px 0 22px}
.hero p,.empty p{color:#526A7A;max-width:720px}
.tag{display:inline-block;background:#E0F1ED;color:#006A64;border-radius:30px;padding:6px 12px;font-size:.75rem;font-weight:700}
.empty{border:1px dashed #ADC3CE;border-radius:18px;background:#FFF;padding:38px 28px;margin:14px 0}
.score{background:#102C3D;border-radius:20px;padding:28px;color:#FFF;margin:12px 0}
.score strong{font-size:3.6rem;display:block;letter-spacing:-.05em}
.track{height:10px;background:#375466;border-radius:10px;overflow:hidden;margin:16px 0}.fill{height:100%;background:#65D5C6}
[data-baseweb="tab-list"]{gap:24px}
@media(max-width:700px){h1{font-size:2.1rem!important}.hero{padding:22px}.empty{padding:24px}}
</style>''', unsafe_allow_html=True)
ready = os.environ.get("ICU_ENABLE_LOCAL_RESEARCH") == "1" and local_artifacts_available()
with st.sidebar:
    st.markdown("## ◈ ICU Observatory")
    st.caption("EXPLAINABLE ML / RESEARCH")
    st.divider()
    st.markdown("### Workspace")
    st.write("Local research" if ready else "Public demonstration")
    st.caption("Observation: ICU hours 0–6\n\nOutcome window: hours 6–30")
    st.divider()
    bands = st.slider("Score bands (%)", 0, 100, (30, 50))
    moderate, high = [v / 100 for v in bands]
    if moderate >= high:
        st.error("Choose two distinct thresholds.")
        st.stop()
    st.caption("Exploratory display settings. Thresholds are not clinically validated and do not change predictions.")
    st.divider()
    st.caption("Research prototype. No clinical alerts or treatment recommendations.")
st.markdown('<div class="eyebrow">RESEARCH WORKSPACE / EARLY DETERIORATION</div>', unsafe_allow_html=True)
st.markdown('''<div class="hero"><span class="tag">RESEARCH PROTOTYPE</span><h1>A clearer view of<br>ICU deterioration risk.</h1><p>Explore the research model’s output, inspect the information behind it, and understand its limits in one transparent workspace.</p></div>''', unsafe_allow_html=True)
st.warning("Not for clinical use. This research prototype has not been clinically validated or approved for patient-care decisions. Do not enter identifiable patient information.")
st.caption("LOCAL RESEARCH MODE · Authorized artifacts loaded" if ready else "PUBLIC MODE · No patient records, model predictions or illustrative scores loaded")
if ready:
    try:
        monitor, x_imp, x_raw = load_local_data()
        model = load_model()
        validate_artifacts(monitor, x_imp, x_raw)
    except Exception:
        st.error("Research artifacts failed validation. Check their schema, completeness and package versions locally.")
        st.stop()
patient, population, about = st.tabs(["01  Patient workspace", "02  Cohort overview", "03  Method & limitations"])

def render_score(p, source):
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid estimator probability.")
    category = risk_category(p, moderate, high)
    st.markdown(f'<div class="score">24-HOUR MODEL ESTIMATE<strong>{p:.1%}</strong>{category} score band<div class="track"><div class="fill" style="width:{p*100:.2f}%"></div></div>{source}</div>', unsafe_allow_html=True)
    st.caption("Bands follow sidebar settings. A lower score does not establish safety; a higher score is not a diagnosis.")

def explain(row, raw):
    st.markdown("### What influenced the underlying model?")
    st.caption("SHAP describes the underlying model before calibration. Contributions are not percentage-point changes in the calibrated score and do not establish medical causality.")
    try:
        with st.spinner("Computing model explanation…"):
            explanation = shap_for_patient(model, row)
            contributions = local_contributions(explanation, raw, 12)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        chart = contributions.sort_values("SHAP value")
        ax.barh(chart["Feature"], chart["SHAP value"], color=["#C17B56" if x > 0 else "#007F78" for x in chart["SHAP value"]])
        ax.axvline(0, color="#A4B4BF", linewidth=.8)
        ax.set_xlabel("SHAP contribution · underlying model output")
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.expander("Inspect explanation values"):
            st.dataframe(contributions, hide_index=True, use_container_width=True)
    except Exception:
        st.info("An explanation is unavailable for this estimator or row. The calibrated score is unchanged.")

with patient:
    left, right = st.columns([1.1, 1.4], gap="large")
    with left:
        st.markdown("### Patient input")
        st.caption("Use prepared research features from the original pipeline. Single vital signs cannot replace the full observation window.")
        if not ready:
            st.selectbox("Research record", [], placeholder="No authorized records loaded", disabled=True)
            st.button("Review model output", disabled=True, use_container_width=True)
            st.info("Prepare the original artifacts and enable local research mode. See the local setup section in README.md.")
        else:
            selected = st.selectbox("Research record", monitor.display_id.tolist())
            row = x_imp.set_index("display_id").loc[[selected]]
            raw = x_raw.set_index("display_id").loc[selected]
            with st.expander("Observation-window inputs", expanded=True):
                snapshot = [dict(Feature=format_value(f, v)[0], Value=format_value(f, v)[1]) for f, v in raw.items()]
                st.dataframe(pd.DataFrame(snapshot), hide_index=True, height=360, use_container_width=True)
    with right:
        if not ready:
            st.markdown('''<div class="empty"><div class="eyebrow">AWAITING RESEARCH INPUT</div><h3>Evidence first. Then an estimate.</h3><p>A risk visualization and explanation will appear here when authorized research artifacts are available.</p><p>No probability is shown without a real model output.</p></div>''', unsafe_allow_html=True)
        else:
            p = float(monitor.set_index("display_id").loc[selected, "calibrated_probability"])
            render_score(p, "Stored calibrated output · original pipeline")
            explain(row, raw)
    with st.expander("Advanced input · run the calibrated estimator"):
        st.write("Upload one complete, already imputed feature row as CSV. Names and order must match the original matrix. No imputation, scaling or feature engineering is added here.")
        bundle_path = DATA_DIR / "model_artifacts.joblib"
        if not ready or not bundle_path.exists():
            st.info("Requires local research mode and model_artifacts.joblib including calibrated_model. Raw-vitals prediction is unavailable because the original pipeline does not export its fitted imputer.")
        else:
            st.download_button("Download schema (no patient values)", json.dumps(list(row.columns), indent=2), "feature_schema.json")
            uploaded = st.file_uploader("Prepared feature row · no identifiers", type=["csv"])
            if uploaded is not None and st.button("Run research inference", type="primary"):
                try:
                    entered = pd.read_csv(uploaded)
                    validate_input(entered, list(row.columns))
                    estimator = joblib.load(bundle_path)["calibrated_model"]
                    positive = list(estimator.classes_).index(1)
                    probability = float(estimator.predict_proba(entered)[0, positive])
                    render_score(probability, "Fresh inference · original calibrated estimator")
                    explain(entered, entered.iloc[0])
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Inference failed. Check artifact compatibility locally.")
with population:
    st.markdown("### Cohort overview")
    st.caption("Retrospective research view. No continuous monitoring or clinical alerts.")
    if not ready:
        for col, title in zip(st.columns(3), ["Loaded records", "Above upper threshold", "Median model score"]):
            col.metric(title, "—")
        st.info("No cohort loaded. Actual artifact scores appear in local research mode.")
    else:
        display = monitor.copy()
        display["Score band"] = display.calibrated_probability.map(lambda p: risk_category(p, moderate, high))
        for col, title, value in zip(st.columns(3), ["Loaded records", "Above upper threshold", "Median model score"], [f"{len(display):,}", f"{(display.calibrated_probability >= high).sum():,}", f"{display.calibrated_probability.median():.1%}"]):
            col.metric(title, value)
        selected_bands = st.multiselect("Show bands", ["Low", "Moderate", "High"], default=["Low", "Moderate", "High"])
        display = display[display["Score band"].isin(selected_bands)].sort_values("calibrated_probability", ascending=False)
        st.dataframe(display, hide_index=True, use_container_width=True, column_config={"calibrated_probability": st.column_config.ProgressColumn("Calibrated score", min_value=0, max_value=1, format="%.3f")})
with about:
    st.markdown("### Understand the model before its output")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Temporal design\nPredictors use the first **6 ICU hours**. The primary outcome window is **hours 6–30**. The research outcome combines mortality, vasopressor requirement and mechanical ventilation.")
        st.markdown("#### Original pipeline\nLogistic regression, random forest and XGBoost comparison; temporal leakage audits; isotonic calibration; SHAP. Preprocessing and evaluation scripts are retained unchanged.")
    with b:
        st.markdown("#### Evidence boundary\nThe source reports retrospective MIMIC-IV v3.1 experiments. Performance is not independently verified here and does not establish prospective clinical utility.")
        st.markdown("#### Intended use\nEducation and authorized retrospective research only. No prospective or external validation is established. Do not use scores for diagnosis, triage, escalation or treatment.")
    st.markdown("#### Privacy & deployment\nPublic mode contains no patient examples. Restricted artifacts require explicit local opt-in. Streamlit caches are shared within a server process; use an authorized single-tenant environment. The interface does not deliberately write uploads or predictions to disk.")
    st.link_button("Original research repository", "https://github.com/anujatj0011/icu-deterioration-prediction")
st.divider()
st.caption("◈ ICU Observatory / Research prototype · Not a medical device · Not for clinical use")
