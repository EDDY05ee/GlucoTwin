from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="GlucoTwin | Digital Twin",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 GlucoTwin")
st.caption(
    "Personalized glucose forecasting prototype | "
    "Digital Twin Challenge 2026"
)

st.warning(
    "SYNTHETIC DEMONSTRATION ONLY. The records and model "
    "results are artificially generated. This system is not "
    "clinically validated and must not guide medical decisions."
)

# Check that the required generated files exist.
required_files = [
    DATA_DIR / "synthetic_ehr.csv",
    DATA_DIR / "synthetic_timeseries.csv",
    DATA_DIR / "forecast_evaluation.csv",
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("Required project data files are missing:")
    for path in missing_files:
        st.write(f"- {path}")
    st.stop()

# Load generated data.
ehr = pd.read_csv(DATA_DIR / "synthetic_ehr.csv")
timeseries = pd.read_csv(
    DATA_DIR / "synthetic_timeseries.csv",
    parse_dates=["timestamp"],
)
evaluation = pd.read_csv(
    DATA_DIR / "forecast_evaluation.csv",
    parse_dates=["timestamp"],
)

st.sidebar.header("Patient selection")

patient_ids = sorted(ehr["patient_id"].unique().tolist())
patient_id = st.sidebar.selectbox("Synthetic patient ID", patient_ids)

profile_rows = ehr[ehr["patient_id"] == patient_id]
patient_history = (
    timeseries[timeseries["patient_id"] == patient_id]
    .sort_values("timestamp")
    .copy()
)
patient_eval = (
    evaluation[evaluation["patient_id"] == patient_id]
    .sort_values("timestamp")
    .copy()
)

if profile_rows.empty or patient_history.empty:
    st.error("No records were found for the selected patient.")
    st.stop()

profile = profile_rows.iloc[0]
latest = patient_history.iloc[-1]

st.sidebar.info(
    "All patient profiles, clinical values, glucose readings, "
    "and wearable-like signals are synthetic."
)

# Patient overview.
st.subheader("1. Synthetic patient profile")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patient ID", patient_id)
c2.metric("Age", f'{int(profile["age_years"])} years')
c3.metric("Diabetes duration", f'{int(profile["diabetes_duration_years"])} years')
c4.metric("Synthetic HbA1c", f'{profile["hba1c_percent"]:.1f}%')

st.caption(
    "These profile values are invented for software testing, "
    "not a real patient's medical record."
)

st.divider()

# Recent historical glucose.
st.subheader("2. Recent glucose history")

recent = patient_history.tail(48)

fig_history = go.Figure()
fig_history.add_trace(
    go.Scatter(
        x=recent["timestamp"],
        y=recent["glucose_mg_dl"],
        mode="lines",
        name="Synthetic glucose",
        line={"width": 2},
    )
)
fig_history.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="Glucose (mg/dL)",
    hovermode="x unified",
    margin={"l": 10, "r": 10, "t": 20, "b": 10},
)
st.plotly_chart(fig_history, use_container_width=True)

g1, g2, g3 = st.columns(3)
g1.metric("Latest synthetic glucose", f'{latest["glucose_mg_dl"]:.1f} mg/dL')
g2.metric("Heart rate", f'{latest["heart_rate_bpm"]:.1f} bpm')
g3.metric("Steps in last 30 min", f'{int(latest["steps_30min"])}')

st.caption(
    "Heart rate and steps are simulated wearable-like signals. "
    "The sleep-period proxy is based on time of day, not measured sleep."
)

# Model evaluation.
st.divider()
st.subheader("3. Held-out model evaluation")

if patient_eval.empty:
    st.info("No evaluation rows are available for this patient.")
else:
    actual = patient_eval["actual_glucose_mg_dl"].to_numpy()
    model_pred = patient_eval["model_prediction_mg_dl"].to_numpy()
    baseline_pred = patient_eval["persistence_prediction_mg_dl"].to_numpy()

    model_mae = mean_absolute_error(actual, model_pred)
    model_rmse = np.sqrt(mean_squared_error(actual, model_pred))
    baseline_mae = mean_absolute_error(actual, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(actual, baseline_pred))

    st.caption(
        "These predictions are from the held-out historical test period. "
        "They are not predictions of the patient's future glucose today."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Random Forest MAE", f"{model_mae:.2f} mg/dL")
    m2.metric("Random Forest RMSE", f"{model_rmse:.2f} mg/dL")
    m3.metric("Baseline MAE", f"{baseline_mae:.2f} mg/dL")
    m4.metric("Baseline RMSE", f"{baseline_rmse:.2f} mg/dL")

    fig_eval = go.Figure()
    fig_eval.add_trace(
        go.Scatter(
            x=patient_eval["timestamp"],
            y=patient_eval["actual_glucose_mg_dl"],
            mode="lines",
            name="Actual synthetic test value",
        )
    )
    fig_eval.add_trace(
        go.Scatter(
            x=patient_eval["timestamp"],
            y=patient_eval["model_prediction_mg_dl"],
            mode="lines",
            name="Random Forest prediction",
        )
    )
    fig_eval.add_trace(
        go.Scatter(
            x=patient_eval["timestamp"],
            y=patient_eval["persistence_prediction_mg_dl"],
            mode="lines",
            name="Persistence baseline",
            line={"dash": "dash"},
        )
    )
    fig_eval.update_layout(
        xaxis_title="Test-period timestamp",
        yaxis_title="Glucose (mg/dL)",
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    comparison = pd.DataFrame(
        {
            "Model": ["Random Forest", "Persistence baseline"],
            "MAE (mg/dL)": [model_mae, baseline_mae],
            "RMSE (mg/dL)": [model_rmse, baseline_rmse],
        }
    )
    st.dataframe(comparison.round(2), hide_index=True, use_container_width=True)

    st.download_button(
        "Download this patient's evaluation CSV",
        data=patient_eval.to_csv(index=False).encode("utf-8"),
        file_name=f"{patient_id}_forecast_evaluation.csv",
        mime="text/csv",
    )

# Data inspection.
with st.expander("View synthetic patient time-series data"):
    st.dataframe(
        patient_history.tail(100),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("4. Prototype status")

st.write("✅ Synthetic historical patient profiles")
st.write("✅ Synthetic glucose and wearable-like time-series")
st.write("✅ Random Forest versus persistence baseline evaluation")
st.write("⬜ Real, appropriately licensed data integration")
st.write("⬜ Stronger temporal and patient-level validation")
st.write("⬜ Future forecasting integration and calibrated uncertainty")
st.write("⬜ Adverse-event prediction with separately evaluated metrics")

st.caption(
    "GlucoTwin | Research prototype | Synthetic data only | "
    "Not for diagnosis or treatment"
)
# --------------------------------------------------
# 5. Synthetic adverse-event evaluation
# --------------------------------------------------
st.divider()
st.subheader("5. High-glucose event model evaluation")

st.caption(
    "Demonstration event: synthetic glucose >= 180 mg/dL "
    "within the next 60 minutes. This is a research label, "
    "not a medical alert or diagnosis."
)

event_path = DATA_DIR / "event_evaluation.csv"

if event_path.exists():
    event_data = pd.read_csv(
        event_path,
        parse_dates=["timestamp"],
    )

    selected_events = event_data[
        event_data["patient_id"] == patient_id
    ].copy()

    if selected_events.empty:
        st.info("No event evaluation rows for this synthetic patient.")
    else:
        actual_event = selected_events["actual_event"].astype(int)
        predicted_event = selected_events["predicted_event"].astype(int)
        probability = selected_events["predicted_probability"]

        tp = int(((actual_event == 1) & (predicted_event == 1)).sum())
        tn = int(((actual_event == 0) & (predicted_event == 0)).sum())
        fp = int(((actual_event == 0) & (predicted_event == 1)).sum())
        fn = int(((actual_event == 1) & (predicted_event == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        false_alarm_rate = (
            fp / (fp + tn) if (fp + tn) else 0.0
        )

        e1, e2, e3 = st.columns(3)
        e1.metric("Precision", f"{precision:.1%}")
        e2.metric("Recall / sensitivity", f"{recall:.1%}")
        e3.metric("False-positive rate", f"{false_alarm_rate:.1%}")

        confusion = pd.DataFrame(
            [
                {"Actual": "No event", "Prediction": "No event", "Count": tn},
                {"Actual": "No event", "Prediction": "Event", "Count": fp},
                {"Actual": "Event", "Prediction": "No event", "Count": fn},
                {"Actual": "Event", "Prediction": "Event", "Count": tp},
            ]
        )

        st.write("**Confusion matrix counts**")
        st.dataframe(
            confusion,
            use_container_width=True,
            hide_index=True,
        )

        event_fig = go.Figure()
        event_fig.add_trace(
            go.Scatter(
                x=selected_events["timestamp"],
                y=probability,
                mode="lines",
                name="Predicted event probability",
            )
        )
        event_fig.add_hline(
            y=0.5,
            line_dash="dash",
            annotation_text="Prototype classification threshold",
        )
        event_fig.update_layout(
            xaxis_title="Held-out test timestamp",
            yaxis_title="Predicted probability",
            yaxis={"range": [0, 1]},
            hovermode="x unified",
        )
        st.plotly_chart(event_fig, use_container_width=True)

        st.dataframe(
            selected_events,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download event evaluation CSV",
            data=selected_events.to_csv(index=False).encode("utf-8"),
            file_name=f"{patient_id}_event_evaluation.csv",
            mime="text/csv",
        )

        st.warning(
            "Synthetic-data evaluation only. The model is not "
            "clinically validated and must not trigger real medical "
            "alerts or guide treatment."
        )
else:
    st.info(
        "Event evaluation file not found. Run "
        "'python train_event_model.py' in the project terminal first."
    )