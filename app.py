from pathlib import Path
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# GLUCOTWIN | RESEARCH INTELLIGENCE DASHBOARD
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="GlucoTwin | Research Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- THEME -----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #102238;
    --navy2: #172F49;
    --teal: #087F8C;
    --teal-light: #E5F5F4;
    --ink: #172B40;
    --muted: #63748A;
    --line: #E0E7EF;
    --surface: #FFFFFF;
    --canvas: #F4F7FB;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--canvas);
    color: var(--ink);
}

header[data-testid="stHeader"] {
    background: rgba(244,247,251,0.96);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.6px;
}

p, label, li, span {
    color: inherit;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102238 0%, #142D45 100%);
    border-right: 1px solid #263E55;
}

[data-testid="stSidebar"] * {
    color: #F0F5FA !important;
}

[data-testid="stSidebar"] label {
    font-weight: 600;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #C7D5E3 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #172B40 !important;
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #1C3D50;
    border: 1px solid #315A6B;
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 19px 21px;
    box-shadow: 0 4px 16px rgba(16,34,56,0.035);
}

[data-testid="stMetricLabel"] {
    color: #63748A !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #102238 !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

[data-testid="stMetricDelta"] {
    color: #087F8C !important;
}

[data-testid="stTabs"] button {
    font-weight: 600;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #087F8C !important;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}

.stButton > button[kind="primary"] {
    background: #087F8C;
    border-color: #087F8C;
    color: white;
}

div[data-testid="stDataFrame"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stPlotlyChart"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 8px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.gt-hero {
    background: linear-gradient(115deg, #102238 0%, #173D55 65%, #087F8C 130%);
    border-radius: 20px;
    padding: 30px 34px;
    color: #FFFFFF;
    margin-bottom: 22px;
    box-shadow: 0 12px 32px rgba(16,34,56,0.10);
}

.gt-hero * {
    color: #FFFFFF !important;
}

.gt-eyebrow {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 700;
    opacity: 0.78;
    margin-bottom: 10px;
}

.gt-title {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(28px, 3vw, 42px);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.2;
}

.gt-subtitle {
    font-size: 15px;
    opacity: 0.85;
    margin-top: 9px;
    line-height: 1.7;
}

.gt-pill {
    display: inline-block;
    border: 1px solid rgba(255,255,255,.28);
    background: rgba(255,255,255,.10);
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
    margin-top: 18px;
}

.gt-section {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #172B40;
    margin: 8px 0 4px;
}

.gt-description {
    font-size: 13px;
    color: #63748A;
    margin-bottom: 18px;
    line-height: 1.7;
}

.gt-card {
    background: #FFFFFF;
    border: 1px solid #E0E7EF;
    border-radius: 15px;
    padding: 19px;
    min-height: 110px;
}

.gt-card-label {
    color: #63748A;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 9px;
}

.gt-card-value {
    color: #102238;
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
}

.gt-card-note {
    color: #718096;
    font-size: 11px;
    margin-top: 6px;
}

.gt-footer {
    margin-top: 35px;
    padding: 18px 0;
    border-top: 1px solid #DCE5EE;
    color: #63748A;
    font-size: 12px;
    line-height: 1.8;
}

.gt-side-brand {
    font-family: 'Manrope', sans-serif;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin-top: 5px;
}

.gt-side-sub {
    color: #B7C8D8 !important;
    font-size: 11px;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-top: 3px;
}

.gt-side-line {
    border-top: 1px solid #345067;
    margin: 22px 0;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .gt-hero {
        padding: 22px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------------- HELPERS ---------------------------

def load_csv(filename, date_columns=None):
    """Read an optional project CSV without crashing the app."""
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame(), f"File not found: data/{filename}"

    try:
        df = pd.read_csv(path, low_memory=False)

        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        return df, None

    except Exception as exc:
        return pd.DataFrame(), f"Could not read {filename}: {exc}"


def humanize(name):
    """Convert technical column names into readable labels."""
    known = {
        "patient_id": "Patient ID",
        "age_years": "Age (years)",
        "diabetes_duration_years": "Diabetes duration (years)",
        "hba1c_percent": "HbA1c (%)",
        "comorbidity_count": "Comorbidity count",
        "data_type": "Data classification",
        "timestamp": "Timestamp",
        "glucose_mg_dl": "Glucose (mg/dL)",
        "heart_rate_bpm": "Heart rate (bpm)",
        "steps_30min": "Steps (last 30 min)",
        "actual_glucose_mg_dl": "Actual glucose (mg/dL)",
        "model_prediction_mg_dl": "Model prediction (mg/dL)",
        "persistence_prediction_mg_dl": "Baseline prediction (mg/dL)",
        "actual_event": "Actual event",
        "predicted_event": "Predicted event",
        "predicted_probability": "Predicted probability",
    }

    if name in known:
        return known[name]

    return str(name).replace("_", " ").strip().title()


def pretty_columns(df):
    result = df.copy()
    result.columns = [humanize(c) for c in result.columns]
    return result


def section(title, description):
    st.markdown(
        f'<div class="gt-section">{html.escape(title)}</div>'
        f'<div class="gt-description">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="gt-card">
          <div class="gt-card-label">{html.escape(str(label))}</div>
          <div class="gt-card-value">{html.escape(str(value))}</div>
          <div class="gt-card-note">{html.escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message):
    st.info(message)


def make_chart_layout(fig, y_title=None, x_title=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="DM Sans, sans-serif",
            color="#34465B",
            size=12,
        ),
        margin=dict(l=18, r=18, t=35, b=18),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    if y_title:
        fig.update_yaxes(title_text=y_title, gridcolor="#EDF1F6")

    if x_title:
        fig.update_xaxes(title_text=x_title, gridcolor="#EDF1F6")

    return fig


def show_table(df, key):
    if df.empty:
        empty_state("No records are available for this selection.")
        return

    st.caption(
        f"{len(df):,} records · {len(df.columns):,} fields"
    )
    st.dataframe(
        pretty_columns(df),
        hide_index=True,
        width="stretch",
        height=360,
        key=key,
    )

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


# ---------------------- LOAD DATA -------------------------

ehr, ehr_error = load_csv("synthetic_ehr.csv")
history, history_error = load_csv(
    "synthetic_timeseries.csv", ["timestamp"]
)
forecast, forecast_error = load_csv(
    "forecast_evaluation.csv", ["timestamp"]
)
events, event_error = load_csv(
    "event_evaluation.csv", ["timestamp"]
)

# ---------------------- SIDEBAR ---------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="gt-side-brand">🧬 GlucoTwin</div>
        <div class="gt-side-sub">Research intelligence</div>
        <div class="gt-side-line"></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace")
    st.caption("Explore the synthetic patient dataset.")

    if not ehr.empty and "patient_id" in ehr.columns:
        patient_ids = sorted(
            ehr["patient_id"].dropna().astype(str).unique().tolist()
        )

        if patient_ids:
            patient_id = st.selectbox(
                "Select synthetic patient",
                patient_ids,
                key="patient_selector",
            )
        else:
            patient_id = None
    else:
        patient_id = None

    st.markdown("---")
    st.markdown("### Dataset status")

    data_status = [
        ("Patient profiles", ehr, ehr_error),
        ("Glucose time series", history, history_error),
        ("Forecast evaluation", forecast, forecast_error),
        ("Event evaluation", events, event_error),
    ]

    for label, df, error in data_status:
        if error:
            st.markdown(f"⚠️ **{label}**")
            st.caption(error)
        else:
            st.markdown(f"✓ **{label}**")
            st.caption(f"{len(df):,} records loaded")

    st.markdown("---")
    st.caption("GlucoTwin Research Prototype")
    st.caption("Synthetic data only · Not for clinical use")


# ---------------------- SELECT PATIENT --------------------

profile = pd.DataFrame()
patient_history = pd.DataFrame()
patient_forecast = pd.DataFrame()
patient_events = pd.DataFrame()

if patient_id is not None:
    if "patient_id" in ehr.columns:
        profile = ehr[
            ehr["patient_id"].astype(str) == patient_id
        ].copy()

    if "patient_id" in history.columns:
        patient_history = history[
            history["patient_id"].astype(str) == patient_id
        ].copy()

    if "timestamp" in patient_history.columns:
        patient_history = patient_history.sort_values("timestamp")

    if "patient_id" in forecast.columns:
        patient_forecast = forecast[
            forecast["patient_id"].astype(str) == patient_id
        ].copy()

    if "timestamp" in patient_forecast.columns:
        patient_forecast = patient_forecast.sort_values("timestamp")

    if "patient_id" in events.columns:
        patient_events = events[
            events["patient_id"].astype(str) == patient_id
        ].copy()

    if "timestamp" in patient_events.columns:
        patient_events = patient_events.sort_values("timestamp")


# ---------------------- HERO ------------------------------

st.markdown(
    """
    <div class="gt-hero">
      <div class="gt-eyebrow">Digital health · Research workspace</div>
      <div class="gt-title">GlucoTwin Intelligence</div>
      <div class="gt-subtitle">
        Explore synthetic glucose patterns, inspect model performance,
        and review event-classification experiments in one workspace.
      </div>
      <div class="gt-pill">● SYNTHETIC RESEARCH ENVIRONMENT</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research demonstration only. All records and results are synthetic. "
    "This application is not clinically validated and must not be used "
    "for diagnosis, treatment, or real medical alerts."
)

if patient_id is None:
    st.error(
        "A patient cannot be selected. Check that "
        "data/synthetic_ehr.csv exists and contains a patient_id column."
    )
    st.stop()


# ---------------------- PATIENT OVERVIEW ------------------

section(
    "Patient overview",
    "A summary of the selected synthetic patient and available observations.",
)

if profile.empty:
    st.warning("No profile was found for this patient.")
else:
    p = profile.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Patient ID", patient_id)

    if "age_years" in profile.columns and pd.notna(p["age_years"]):
        col2.metric("Age", f"{p['age_years']:g} years")
    else:
        col2.metric("Age", "Not available")

    if (
        "diabetes_duration_years" in profile.columns
        and pd.notna(p["diabetes_duration_years"])
    ):
        col3.metric(
            "Diabetes duration",
            f"{p['diabetes_duration_years']:g} years",
        )
    else:
        col3.metric("Diabetes duration", "Not available")

    if (
        "hba1c_percent" in profile.columns
        and pd.notna(p["hba1c_percent"])
    ):
        col4.metric("Synthetic HbA1c", f"{p['hba1c_percent']:.1f}%")
    else:
        col4.metric("Synthetic HbA1c", "Not available")

    with st.expander("View full synthetic patient profile"):
        show_table(profile, f"{patient_id}_profile")

st.markdown("---")


# ---------------------- NAVIGATION ------------------------

tab_history, tab_forecast, tab_events, tab_data, tab_about = st.tabs(
    [
        "📈  Glucose history",
        "🎯  Forecast evaluation",
        "⚡  Event evaluation",
        "▤  Data explorer",
        "ⓘ  About",
    ]
)


# =========================================================
# TAB 1: GLUCOSE HISTORY
# =========================================================

with tab_history:
    section(
        "Glucose timeline",
        "Explore simulated glucose observations and wearable-like signals.",
    )

    if patient_history.empty:
        empty_state(
            "No time-series records were found for this patient. "
            "Check data/synthetic_timeseries.csv."
        )
    elif "glucose_mg_dl" not in patient_history.columns:
        st.error(
            "The time-series file is missing the glucose_mg_dl column."
        )
        show_table(patient_history.tail(50), f"{patient_id}_history")
    else:
        window = st.selectbox(
            "Timeline window",
            options=[24, 48, 96, 168, 336],
            index=1,
            format_func=lambda n: f"Last {n} records",
        )

        recent = patient_history.tail(window).copy()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recent["timestamp"]
                if "timestamp" in recent.columns
                else list(range(len(recent))),
                y=recent["glucose_mg_dl"],
                mode="lines",
                name="Synthetic glucose",
                line=dict(color="#087F8C", width=3),
                fill="tozeroy",
                fillcolor="rgba(8,127,140,0.08)",
                hovertemplate="%{y:.1f} mg/dL<extra></extra>",
            )
        )

        make_chart_layout(
            fig,
            "Glucose (mg/dL)",
            "Timestamp" if "timestamp" in recent.columns else "Record",
        )

        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

        latest = recent.iloc[-1]

        a, b, c = st.columns(3)

        latest_glucose = latest["glucose_mg_dl"]
        a.metric(
            "Latest synthetic glucose",
            f"{latest_glucose:.1f} mg/dL"
            if pd.notna(latest_glucose)
            else "Not available",
        )

        if "heart_rate_bpm" in recent.columns:
            hr = recent.iloc[-1]["heart_rate_bpm"]
            b.metric(
                "Heart rate",
                f"{hr:.1f} bpm" if pd.notna(hr) else "Not available",
            )
        else:
            b.metric("Heart rate", "Not available")

        if "steps_30min" in recent.columns:
            steps = recent.iloc[-1]["steps_30min"]
            c.metric(
                "Steps in last 30 min",
                f"{steps:,.0f}" if pd.notna(steps) else "Not available",
            )
        else:
            c.metric("Steps in last 30 min", "Not available")

        st.caption(
            "All values are simulated. Wearable-like signals are not "
            "measurements from a real device."
        )

        with st.expander("Inspect recent records"):
            show_table(recent.tail(100), f"{patient_id}_recent_history")


# =========================================================
# TAB 2: FORECAST EVALUATION
# =========================================================

with tab_forecast:
    section(
        "Forecast model performance",
        "Compare held-out model predictions with a persistence baseline.",
    )

    required_forecast = [
        "actual_glucose_mg_dl",
        "model_prediction_mg_dl",
        "persistence_prediction_mg_dl",
    ]

    if patient_forecast.empty:
        empty_state(
            "No forecast evaluation rows are available for this patient. "
            "Check data/forecast_evaluation.csv and the patient ID."
        )
    elif not all(c in patient_forecast.columns for c in required_forecast):
        st.error(
            "The forecast CSV does not contain all required evaluation "
            "columns. Expected: actual_glucose_mg_dl, "
            "model_prediction_mg_dl, persistence_prediction_mg_dl."
        )
        show_table(patient_forecast, f"{patient_id}_forecast")
    else:
        evaluation_clean = patient_forecast.copy()

        for col in required_forecast:
            evaluation_clean[col] = pd.to_numeric(
                evaluation_clean[col], errors="coerce"
            )

        evaluation_clean = evaluation_clean.dropna(
            subset=required_forecast
        )

        if evaluation_clean.empty:
            empty_state(
                "No complete numeric prediction rows are available "
                "for metric calculation."
            )
        else:
            actual = evaluation_clean["actual_glucose_mg_dl"].to_numpy()
            prediction = evaluation_clean[
                "model_prediction_mg_dl"
            ].to_numpy()
            baseline = evaluation_clean[
                "persistence_prediction_mg_dl"
            ].to_numpy()

            model_mae = mean_absolute_error(actual, prediction)
            model_rmse = np.sqrt(
                mean_squared_error(actual, prediction)
            )
            baseline_mae = mean_absolute_error(actual, baseline)
            baseline_rmse = np.sqrt(
                mean_squared_error(actual, baseline)
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model MAE", f"{model_mae:.2f} mg/dL")
            c2.metric("Model RMSE", f"{model_rmse:.2f} mg/dL")
            c3.metric("Baseline MAE", f"{baseline_mae:.2f} mg/dL")
            c4.metric("Baseline RMSE", f"{baseline_rmse:.2f} mg/dL")

            fig = go.Figure()

            x = (
                evaluation_clean["timestamp"]
                if "timestamp" in evaluation_clean.columns
                else list(range(len(evaluation_clean)))
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=actual,
                    name="Actual synthetic value",
                    mode="lines",
                    line=dict(color="#172B40", width=2.4),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=prediction,
                    name="Random Forest prediction",
                    mode="lines",
                    line=dict(color="#087F8C", width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=baseline,
                    name="Persistence baseline",
                    mode="lines",
                    line=dict(color="#9A82C0", width=2, dash="dash"),
                )
            )

            make_chart_layout(
                fig,
                "Glucose (mg/dL)",
                "Timestamp" if "timestamp" in evaluation_clean.columns
                else "Test record",
            )

            fig.update_layout(height=440)
            st.plotly_chart(fig, width="stretch")

            comparison = pd.DataFrame({
                "Model": ["Random Forest", "Persistence baseline"],
                "MAE (mg/dL)": [model_mae, baseline_mae],
                "RMSE (mg/dL)": [model_rmse, baseline_rmse],
            })

            st.markdown("#### Model comparison")
            st.dataframe(
                comparison.round(3),
                hide_index=True,
                width="stretch",
            )

            st.download_button(
                "Download forecast evaluation CSV",
                data=evaluation_clean.to_csv(index=False).encode("utf-8"),
                file_name=f"{patient_id}_forecast_evaluation.csv",
                mime="text/csv",
            )

            st.caption(
                "These are retrospective held-out evaluation results, "
                "not predictions of current or future real-world glucose."
            )


# =========================================================
# TAB 3: EVENT EVALUATION
# =========================================================

with tab_events:
    section(
        "High-glucose event evaluation",
        "Review the synthetic classifier's predictions and classification metrics.",
    )

    required_events = [
        "actual_event",
        "predicted_event",
        "predicted_probability",
    ]

    if events.empty:
        empty_state(
            "The event evaluation file is unavailable. If needed, run "
            "python train_event_model.py in your project terminal."
        )
    elif patient_events.empty:
        empty_state(
            "No event evaluation rows were found for this patient."
        )
    elif not all(c in patient_events.columns for c in required_events):
        st.error(
            "The event CSV is missing required columns: "
            "actual_event, predicted_event, predicted_probability."
        )
        show_table(patient_events, f"{patient_id}_events")
    else:
        ev = patient_events.copy()

        for col in required_events:
            ev[col] = pd.to_numeric(ev[col], errors="coerce")

        ev = ev.dropna(subset=required_events)

        if ev.empty:
            empty_state("No complete event evaluation rows are available.")
        else:
            actual_event = ev["actual_event"].astype(int)
            predicted_event = ev["predicted_event"].astype(int)

            tp = int(
                ((actual_event == 1) & (predicted_event == 1)).sum()
            )
            tn = int(
                ((actual_event == 0) & (predicted_event == 0)).sum()
            )
            fp = int(
                ((actual_event == 0) & (predicted_event == 1)).sum()
            )
            fn = int(
                ((actual_event == 1) & (predicted_event == 0)).sum()
            )

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            false_positive_rate = (
                fp / (fp + tn) if fp + tn else 0.0
            )

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Precision", f"{precision:.1%}")
            e2.metric("Recall / sensitivity", f"{recall:.1%}")
            e3.metric("Specificity", f"{specificity:.1%}")
            e4.metric("F1 score", f"{f1:.3f}")

            st.markdown("#### Confusion matrix")

            confusion = pd.DataFrame(
                [
                    [tn, fp],
                    [fn, tp],
                ],
                index=["Actual: no event", "Actual: event"],
                columns=["Predicted: no event", "Predicted: event"],
            )

            st.dataframe(confusion, width="stretch")

            st.caption(
                f"True negatives: {tn} · False positives: {fp} · "
                f"False negatives: {fn} · True positives: {tp} · "
                f"False-positive rate: {false_positive_rate:.1%}"
            )

            fig = go.Figure()

            x = (
                ev["timestamp"]
                if "timestamp" in ev.columns
                else list(range(len(ev)))
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ev["predicted_probability"],
                    name="Predicted event probability",
                    mode="lines",
                    line=dict(color="#087F8C", width=2.5),
                )
            )

            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="#9A82C0",
                annotation_text="Classification threshold",
            )

            make_chart_layout(
                fig,
                "Predicted probability",
                "Timestamp" if "timestamp" in ev.columns
                else "Test record",
            )

            fig.update_yaxes(range=[0, 1])
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

            st.download_button(
                "Download event evaluation CSV",
                data=ev.to_csv(index=False).encode("utf-8"),
                file_name=f"{patient_id}_event_evaluation.csv",
                mime="text/csv",
            )

            st.warning(
                "This classifier uses synthetic data and is not "
                "clinically validated. It must not trigger real medical "
                "alerts or guide treatment."
            )


# =========================================================
# TAB 4: DATA EXPLORER
# =========================================================

with tab_data:
    section(
        "Data explorer",
        "Inspect the underlying datasets using readable field names and export records.",
    )

    datasets = {
        "Patient profiles": ehr,
        "Glucose time series": history,
        "Forecast evaluation": forecast,
        "Event evaluation": events,
    }

    dataset_name = st.selectbox(
        "Choose dataset",
        list(datasets.keys()),
        key="explorer_dataset",
    )

    selected_df = datasets[dataset_name].copy()

    if selected_df.empty:
        empty_state(
            f"No data is available for {dataset_name.lower()}. "
            "Check the corresponding CSV in the data folder."
        )
    else:
        filter_patient = st.checkbox(
            "Filter by selected patient",
            value=True,
            key="explorer_filter_patient",
        )

        if (
            filter_patient
            and "patient_id" in selected_df.columns
            and patient_id is not None
        ):
            selected_df = selected_df[
                selected_df["patient_id"].astype(str) == patient_id
            ]

        st.caption(
            f"Dataset: {dataset_name} · "
            f"{len(selected_df):,} rows · "
            f"{len(selected_df.columns):,} columns"
        )

        # Friendly names are for display only. Original column names
        # remain unchanged in downloaded CSV files.
        st.dataframe(
            pretty_columns(selected_df),
            hide_index=True,
            width="stretch",
            height=430,
        )

        st.download_button(
            "Download selected dataset",
            data=selected_df.to_csv(index=False).encode("utf-8"),
            file_name=(
                dataset_name.lower()
                .replace(" ", "_")
                + ".csv"
            ),
            mime="text/csv",
            key="download_explorer",
        )

        with st.expander("Technical field reference"):
            field_reference = pd.DataFrame({
                "Display name": [humanize(c) for c in selected_df.columns],
                "Original column": list(selected_df.columns),
                "Data type": [
                    str(selected_df[c].dtype)
                    for c in selected_df.columns
                ],
            })

            st.dataframe(
                field_reference,
                hide_index=True,
                width="stretch",
            )


# =========================================================
# TAB 5: ABOUT
# =========================================================

with tab_about:
    section(
        "About GlucoTwin",
        "Project scope, research boundaries, and implementation status.",
    )

    st.markdown("### Project purpose")

    st.write(
        "GlucoTwin is a software research prototype that combines "
        "synthetic patient profiles, simulated glucose time series, "
        "retrospective forecast evaluation, and a separate "
        "high-glucose event classifier."
    )

    st.markdown("### Available components")

    components = [
        ("Synthetic patient profiles", not ehr.empty),
        ("Synthetic glucose time series", not history.empty),
        ("Forecast evaluation dataset", not forecast.empty),
        ("Event evaluation dataset", not events.empty),
    ]

    for label, available in components:
        st.write(f"{'✓' if available else '—'}  {label}")

    st.markdown("### Important limitations")

    st.write(
        "- Patient records and measurements are synthetic.\n"
        "- Retrospective model metrics do not establish clinical performance.\n"
        "- Event classification is a research demonstration, not a medical alert.\n"
        "- No real patient data integration or clinical validation is implied."
    )

    st.markdown("### Project data files")

    st.code(
        "GlucoTwin/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "├── train_model.py\n"
        "├── train_event_model.py\n"
        "└── data/\n"
        "    ├── synthetic_ehr.csv\n"
        "    ├── synthetic_timeseries.csv\n"
        "    ├── forecast_evaluation.csv\n"
        "    └── event_evaluation.csv",
        language="text",
    )


# ---------------------- FOOTER ----------------------------

st.markdown(
    """
    <div class="gt-footer">
      <strong>GlucoTwin · Research Intelligence</strong><br>
      Digital health research prototype · Synthetic data only<br>
      Not for diagnosis, treatment decisions, or real-world medical alerts.
    </div>
    """,
    unsafe_allow_html=True,
)