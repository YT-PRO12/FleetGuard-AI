from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

st.set_page_config(page_title="FleetGuard AI", page_icon="🚗", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())

model = load_model()
df = load_data()
meta = load_metrics()

st.sidebar.title("FleetGuard AI")
page = st.sidebar.radio("Go to", ["Overview", "Predict Failure Risk", "Model Results"])
st.sidebar.caption("AI4I 2020 Predictive Maintenance Dataset")

if page == "Overview":
    st.title("FleetGuard AI")
    st.subheader("Predictive Maintenance Risk Prediction")
    st.write(
        "This project uses classical Machine Learning to predict whether industrial equipment "
        "is likely to fail from six operating parameters."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Observed Failures", int(df["Machine failure"].sum()))
    c3.metric("Failure Rate", f"{df['Machine failure'].mean()*100:.2f}%")

    st.markdown("### Input Features")
    st.write("Product Type, Air Temperature, Process Temperature, Rotational Speed, Torque and Tool Wear")
    st.markdown("### Simple ML Workflow")
    st.info("Dataset → Cleaning → Train/Test Split → Preprocessing → Model Training → Evaluation → Prediction")

elif page == "Predict Failure Risk":
    st.title("Predict Failure Risk")
    st.write("Enter operating values and click **Predict**.")

    c1, c2 = st.columns(2)
    with c1:
        product_type = st.selectbox("Product Type", ["L", "M", "H"])
        air_temp = st.number_input("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.number_input("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)
    with c2:
        rpm = st.number_input("Rotational Speed (rpm)", 1000, 3000, 1500, 10)
        torque = st.number_input("Torque (Nm)", 0.0, 80.0, 40.0, 0.5)
        tool_wear = st.number_input("Tool Wear (min)", 0, 260, 100, 1)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "Type": product_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }])
        probability = float(model.predict_proba(row)[0, 1])
        prediction = int(model.predict(row)[0])

        st.markdown("### Prediction Result")
        st.metric("Failure Probability", f"{probability*100:.1f}%")
        if prediction == 1:
            st.error("Maintenance Risk: FAILURE RISK DETECTED")
            st.write("Recommendation: Inspect the equipment and schedule preventive maintenance.")
        else:
            st.success("Maintenance Risk: NO IMMEDIATE FAILURE DETECTED")
            st.write("Recommendation: Continue routine monitoring and scheduled maintenance.")

elif page == "Model Results":
    st.title("Model Results")
    st.write("Four classical classification algorithms were compared using the same train/test split.")
    rows = []
    for name, m in meta["results"].items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
        })
    result_df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    st.dataframe(result_df, use_container_width=True, hide_index=True)
    st.success(f"Selected Model: {meta['selected_model']} (highest failure-class F1 score in this simplified version)")
    st.caption("Because failures are rare, F1 and recall are more informative than accuracy alone.")
