"""HexaCare Streamlit application for 30-day readmission-risk prediction."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BUNDLE_PATH = Path("model/hexacare_bundle.joblib")
DISCLAIMER = "Educational decision-support prototype only; it does not replace clinical judgment."

@st.cache_resource
def load_bundle(path):
    if not path.exists(): return None, f"Model bundle not found at `{path}`."
    try:
        bundle = joblib.load(path)
        required = {"pipeline", "decision_threshold", "feature_names", "feature_schema"}
        missing = required.difference(bundle)
        if missing: raise ValueError(f"Bundle is missing: {sorted(missing)}")
        return bundle, None
    except Exception as exc: return None, f"Could not load the model bundle: {exc}"

def friendly_name(name):
    labels = {"time_in_hospital":"Time in hospital (days)","num_lab_procedures":"Number of laboratory procedures","num_procedures":"Number of procedures","num_medications":"Number of medications","number_outpatient":"Previous outpatient visits","number_emergency":"Previous emergency visits","number_inpatient":"Previous inpatient visits","number_diagnoses":"Number of diagnoses","A1Cresult":"HbA1c result","max_glu_serum":"Maximum glucose serum result","diabetesMed":"Diabetes medication prescribed","change":"Diabetes medication changed","diag_1_group":"Primary diagnosis group","diag_2_group":"Secondary diagnosis group","diag_3_group":"Additional diagnosis group"}
    return labels.get(name, name.replace("_", " ").title())

def collect_inputs(schema, feature_names):
    values = {}
    tabs = st.tabs(["Patient & encounter", "Clinical", "Medications"])
    encounter = {"race","gender","age","time_in_hospital","admission_type","admission_source","discharge_disposition","number_outpatient","number_emergency","number_inpatient"}
    clinical = {"num_lab_procedures","num_procedures","num_medications","number_diagnoses","max_glu_serum","A1Cresult","diag_1_group","diag_2_group","diag_3_group"}
    for name in feature_names:
        tab = tabs[0] if name in encounter else tabs[1] if name in clinical else tabs[2]
        spec = schema[name]
        with tab:
            if spec["kind"] == "numeric":
                values[name] = st.number_input(friendly_name(name), min_value=int(spec["min"]), max_value=int(spec["max"]), value=int(round(spec["default"])), step=1, key=name)
            else:
                options = spec["options"]
                values[name] = st.selectbox(friendly_name(name), options=options, index=options.index(spec["default"]), key=name)
    return values

def predict(bundle, values):
    frame = pd.DataFrame([values], columns=bundle["feature_names"])
    pipeline = bundle["pipeline"]
    classes = list(pipeline.classes_)
    if 1 not in classes: raise ValueError(f"Positive class 1 is absent; classes: {classes}")
    probability = float(pipeline.predict_proba(frame)[0, classes.index(1)])
    threshold = float(bundle["decision_threshold"])
    return probability, int(probability >= threshold)

st.set_page_config(page_title="HexaCare", page_icon="🏥", layout="wide")
st.title("🏥 HexaCare")
st.subheader("30-Day Hospital Readmission Risk — Patients with Diabetes")
st.info(DISCLAIMER)
bundle, load_error = load_bundle(BUNDLE_PATH)
if load_error:
    st.error(load_error); st.stop()
st.caption(f"Model: {bundle.get('model_name', 'final pipeline')} · Decision threshold: {bundle['decision_threshold']:.4f}")
with st.form("prediction_form"):
    patient_input = collect_inputs(bundle["feature_schema"], bundle["feature_names"])
    submitted = st.form_submit_button("Predict readmission risk", type="primary", use_container_width=True)
if submitted:
    try:
        probability, prediction = predict(bundle, patient_input)
        st.header("Prediction result")
        if prediction: st.error("Higher risk: flagged for possible readmission within 30 days")
        else: st.success("Lower risk: not flagged at the selected decision threshold")
        st.metric("Model score", f"{probability:.1%}")
        st.progress(max(0.0, min(1.0, probability)))
        st.caption(f"Classification uses threshold {bundle['decision_threshold']:.4f}, not 0.50. The score is not a clinically calibrated probability unless calibration is validated separately.")
    except Exception as exc: st.error(f"Prediction failed: {exc}")
