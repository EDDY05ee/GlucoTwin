from pathlib import Path
import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# GlucoTwin | Research Intelligence Dashboard
# Defensive UI for synthetic research data; not a medical tool.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="GlucoTwin | Research Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ DESIGN SYSTEM ------------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--navy:#102238;--navy2:#18364d;--teal:#087f8c;--teal2:#0b9aa5;--ink:#172b40;--muted:#64748b;--line:#e2e8f0;--canvas:#f5f8fc;--white:#fff;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp,[data-testid="stAppViewContainer"]{background:var(--canvas)!important;color:var(--ink);}
[data-testid="stMain"]{background:var(--canvas)!important;}
.block-container{max-width:1540px;padding-top:1.35rem;padding-bottom:3rem;}
h1,h2,h3,h4{font-family:'Manrope',sans-serif!important;color:var(--ink)!important;letter-spacing:-.025em;}
h1{font-weight:800!important;} h2,h3{font-weight:750!important;}
p,li{line-height:1.65;}
[data-testid="stSidebar"]{background:linear-gradient(165deg,#102238 0%,#173b53 62%,#0d5963 130%)!important;border-right:1px solid #28465b;}
/* Sidebar text stays light, except form controls that use a white background. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#edf4fa!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#d0deea!important;}
[data-testid="stSidebar"] [data-testid="stSelectbox"] label{color:#f4f8fc!important;font-weight:650!important;}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#ffffff!important;border:1px solid #c5d3e0!important;border-radius:10px!important;min-height:46px;box-shadow:0 1px 2px rgba(0,0,0,.08);}
/* Force the selected patient ID, placeholder, and dropdown icon to high-contrast ink. */
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] svg{color:#102238!important;fill:#102238!important;opacity:1!important;}
[data-testid="stSidebar"] [data-baseweb="select"] [aria-selected="true"],
[data-testid="stSidebar"] [data-baseweb="select"] [role="option"]{color:#102238!important;}
/* Streamlit renders select menus in a portal, outside the sidebar DOM. */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"]{background:#ffffff!important;color:#102238!important;}
[data-baseweb="popover"] [role="option"] *{color:#102238!important;opacity:1!important;}
[data-baseweb="popover"] [role="option"][aria-selected="true"]{background:#e6f5f5!important;color:#102238!important;}
[data-testid="stSidebar"] hr{border-color:#ffffff35!important;}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:15px;padding:16px 18px;box-shadow:0 4px 16px rgba(16,34,56,.035);min-height:112px;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.82rem!important;font-weight:650!important;}
[data-testid="stMetricValue"]{color:var(--navy)!important;font-family:'Manrope',sans-serif;font-weight:800!important;letter-spacing:-.03em;}
[data-testid="stMetricDelta"]{font-size:.78rem!important;}
.stButton>button,.stDownloadButton>button{border-radius:10px;min-height:42px;font-weight:650;border:1px solid #d9e2eb;}
.stButton>button[kind="primary"]{background:var(--teal);border-color:var(--teal);color:white;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--teal);color:var(--teal);}
[data-testid="stDataFrame"]{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:15px;padding:7px;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
[data-testid="stTabs"] button[data-baseweb="tab"]{padding:12px 15px;border-radius:9px 9px 0 0;font-weight:650;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--teal)!important;border-bottom-color:var(--teal)!important;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#fff;}
.gt-brand{font-family:'Manrope',sans-serif;font-weight:800;font-size:23px;letter-spacing:-.8px;}
.gt-brand-sub{font-size:10px;text-transform:uppercase;letter-spacing:1.8px;color:#b8c9d7;margin-top:1px;}
.gt-hero{position:relative;overflow:hidden;background:linear-gradient(115deg,#102238 0%,#173d55 63%,#087f8c 135%);border-radius:20px;padding:30px 34px;color:white;margin-bottom:18px;box-shadow:0 14px 35px rgba(16,34,56,.12);}
.gt-hero:after{content:'';position:absolute;width:250px;height:250px;border:1px solid #ffffff18;border-radius:50%;right:-65px;top:-95px;box-shadow:0 0 0 28px #ffffff08,0 0 0 58px #ffffff05;}
.gt-hero *{color:#fff!important;position:relative;z-index:1;}
.gt-eyebrow{font-size:10px;letter-spacing:2px;text-transform:uppercase;font-weight:800;opacity:.8;}
.gt-title{font-family:'Manrope',sans-serif;font-size:clamp(28px,3vw,42px);font-weight:800;letter-spacing:-1.5px;line-height:1.16;margin-top:9px;}
.gt-subtitle{font-size:14px;opacity:.88;margin-top:10px;line-height:1.7;max-width:780px;}
.gt-pill{display:inline-block;border:1px solid #ffffff4c;background:#ffffff16;padding:6px 11px;border-radius:24px;font-size:10px;letter-spacing:.7px;font-weight:800;margin-top:17px;}
.gt-section{font-family:'Manrope',sans-serif;font-size:21px;font-weight:800;color:var(--ink);letter-spacing:-.6px;margin:4px 0 2px;}
.gt-description{font-size:13px;color:var(--muted);margin-bottom:15px;line-height:1.65;}
.gt-mini-label{font-size:10px;text-transform:uppercase;letter-spacing:1.4px;font-weight:800;color:#7a8ca0;}
.gt-footer{margin-top:30px;padding:17px 0;border-top:1px solid #dce5ee;color:#64748b;font-size:12px;line-height:1.8;}
.gt-status{padding:12px 14px;background:#fff;border:1px solid #d7e0e8;border-radius:12px;margin:6px 0;}
/* Keep status-card text readable even inside the dark sidebar. */
.gt-status b{color:#172b40!important;}
.gt-status .gt-status-detail{color:#475569!important;}
.gt-status .gt-status-state{font-weight:750;font-size:12px;}
@media(max-width:768px){.block-container{padding-left:1rem;padding-right:1rem}.gt-hero{padding:23px}.gt-title{font-size:29px}[data-testid="stMetric"]{padding:13px;}}
</style>
""", unsafe_allow_html=True)

# -------------------------------- HELPERS ----------------------------------
def load_csv(filename, date_columns=()):
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame(), f"Missing file: data/{filename}"
    try:
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            return frame, f"data/{filename} exists but contains no data rows."
        frame.columns = [str(c).strip() for c in frame.columns]
        for col in date_columns:
            found = find_col(frame, [col])
            if found:
                frame[found] = pd.to_datetime(frame[found], errors="coerce")
        return frame, None
    except Exception as exc:
        return pd.DataFrame(), f"Could not read data/{filename}: {type(exc).__name__}: {exc}"


def find_col(df, candidates):
    if df is None or df.empty:
        return None
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for name in candidates:
        key = re.sub(r"[^a-z0-9]", "", str(name).lower())
        if key in lookup:
            return lookup[key]
    return None


def clean_ids(df):
    col = find_col(df, ["patient_id", "patient", "subject_id", "subject"])
    if df.empty or not col:
        return []
    vals = df[col].dropna().astype(str).str.strip()
    return sorted(v for v in vals.unique().tolist() if v and v.lower() not in {"nan", "none", "null"})


def subset_patient(df, patient_id):
    if df is None or df.empty or not patient_id:
        return pd.DataFrame() if df is None else df.iloc[0:0].copy()
    col = find_col(df, ["patient_id", "patient", "subject_id", "subject"])
    if not col:
        return df.iloc[0:0].copy()
    ids = df[col].astype(str).str.strip()
    return df.loc[ids.eq(str(patient_id).strip())].copy()


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_number(value):
    try:
        n = float(value)
        return n if np.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def pretty_df(df):
    out = df.copy()
    out.columns = [str(c).replace("_", " ").strip().title() for c in out.columns]
    return out


def section(title, description=""):
    st.markdown(f'<div class="gt-section">{html.escape(title)}</div><div class="gt-description">{html.escape(description)}</div>', unsafe_allow_html=True)


def show_table(df, key, height=340):
    if df is None or df.empty:
        st.info("There are no records to display for this selection.")
        return
    st.caption(f"{len(df):,} records · {len(df.columns):,} fields")
    st.dataframe(pretty_df(df), hide_index=True, use_container_width=True, height=height, key=f"table_{key}")
    st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"{key}.csv", mime="text/csv", key=f"download_{key}")


def plot_style(fig, y_title, x_title):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#34465b", size=12),
        margin=dict(l=15, r=15, t=35, b=15), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text=y_title, gridcolor="#edf1f6", zerolinecolor="#e6edf4")
    fig.update_xaxes(title_text=x_title, gridcolor="#edf1f6")
    return fig


def status_card(label, df, error):
    state = "Ready" if error is None and not df.empty else "Needs attention"
    color = "#087f8c" if state == "Ready" else "#b7791f"
    detail = f"{len(df):,} rows loaded" if state == "Ready" else (error or "No rows available")
    st.markdown(
        f'<div class="gt-status">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:center">'
        f'<b>{html.escape(label)}</b>'
        f'<span class="gt-status-state" style="color:{color}">● {state}</span>'
        f'</div>'
        f'<div class="gt-status-detail" style="margin-top:7px">{html.escape(detail)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def timestamp_x(df):
    return find_col(df, ["timestamp", "datetime", "date_time", "date", "time"])


def binary_values(series):
    """Normalize common binary representations to 0/1, preserving unknowns as NaN."""
    if pd.api.types.is_numeric_dtype(series):
        nums = pd.to_numeric(series, errors="coerce")
        return nums.where(nums.isin([0, 1]))
    mapping = {
        "0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1,
        "negative": 0, "positive": 1, "normal": 0, "event": 1,
        "no_event": 0, "event_detected": 1, "high": 1,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).astype(float)


# -------------------------------- LOAD DATA -------------------------------
ehr, ehr_error = load_csv("synthetic_ehr.csv")
history, history_error = load_csv("synthetic_timeseries.csv", ["timestamp", "datetime", "date"])
forecast, forecast_error = load_csv("forecast_evaluation.csv", ["timestamp", "datetime", "date"])
events, events_error = load_csv("event_evaluation.csv", ["timestamp", "datetime", "date"])

# -------------------------------- SIDEBAR ---------------------------------
with st.sidebar:
    st.markdown('<div class="gt-brand">🧬 GlucoTwin</div><div class="gt-brand-sub">Research intelligence platform</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Workspace")
    st.caption("Select a synthetic profile to explore the available records.")

    all_ids = sorted(set(clean_ids(ehr) + clean_ids(history) + clean_ids(forecast) + clean_ids(events)))
    if all_ids:
        default_id = st.session_state.get("gt_patient", all_ids[0])
        if default_id not in all_ids:
            default_id = all_ids[0]
        selected_patient = st.selectbox("Synthetic patient", options=all_ids, index=all_ids.index(default_id), key="gt_patient", help="Changing this selection filters patient-level views where patient IDs are available.")
    else:
        selected_patient = None
        st.error("No patient IDs were found in the available CSV files.")

    st.markdown("---")
    st.markdown("### Data sources")
    status_card("Patient profiles", ehr, ehr_error)
    status_card("Glucose time series", history, history_error)
    status_card("Forecast evaluation", forecast, forecast_error)
    status_card("Event evaluation", events, events_error)
    st.markdown("---")
    st.caption("Synthetic data only · Research prototype")

# -------------------------------- HEADER -----------------------------------
st.markdown("""
<div class="gt-hero">
  <div class="gt-eyebrow">Digital health · Research workspace</div>
  <div class="gt-title">GlucoTwin Intelligence</div>
  <div class="gt-subtitle">A unified workspace to explore simulated glucose trends, review retrospective forecast performance, and inspect event-classification experiments.</div>
  <div class="gt-pill">● SYNTHETIC RESEARCH ENVIRONMENT</div>
</div>
""", unsafe_allow_html=True)

st.warning("Research demonstration only. All records and results are synthetic. This application is not clinically validated and must not be used for diagnosis, treatment, real-time alerts, or treatment decisions.")

if selected_patient:
    profile = subset_patient(ehr, selected_patient)
    patient_history = subset_patient(history, selected_patient)
    patient_forecast = subset_patient(forecast, selected_patient)
    patient_events = subset_patient(events, selected_patient)
else:
    profile = pd.DataFrame()
    patient_history = pd.DataFrame()
    patient_forecast = pd.DataFrame()
    patient_events = pd.DataFrame()

for frame in (patient_history, patient_forecast, patient_events):
    if not frame.empty:
        dt = timestamp_x(frame)
        if dt:
            frame.sort_values(dt, inplace=True, na_position="last")

# ------------------------------ PATIENT OVERVIEW --------------------------
section("Patient overview", "A quick snapshot of the selected synthetic profile and the records available for analysis.")
if not selected_patient:
    st.info("No patient ID is available yet. Check the files in the data/ folder and ensure they contain a patient_id column.")
else:
    p = profile.iloc[0] if not profile.empty else pd.Series(dtype=object)
    age_col = find_col(profile, ["age_years", "age"])
    duration_col = find_col(profile, ["diabetes_duration_years", "diabetes_duration"])
    hba1c_col = find_col(profile, ["hba1c_percent", "hba1c", "hb_a1c"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected profile", selected_patient)
    age = safe_number(p.get(age_col)) if age_col else None
    duration = safe_number(p.get(duration_col)) if duration_col else None
    hba1c = safe_number(p.get(hba1c_col)) if hba1c_col else None
    c2.metric("Age", f"{age:.0f} years" if age is not None else "Not available")
    c3.metric("Diabetes duration", f"{duration:.1f} years" if duration is not None else "Not available")
    c4.metric("Synthetic HbA1c", f"{hba1c:.1f}%" if hba1c is not None else "Not available")

    d1, d2, d3 = st.columns(3)
    d1.metric("Glucose observations", f"{len(patient_history):,}")
    d2.metric("Forecast evaluation rows", f"{len(patient_forecast):,}")
    d3.metric("Event evaluation rows", f"{len(patient_events):,}")
    if not profile.empty:
        with st.expander("View complete synthetic profile"):
            show_table(profile, f"{selected_patient}_profile", height=220)
    elif not ehr.empty:
        st.info(f"{selected_patient} appears in another dataset, but no matching profile row was found in synthetic_ehr.csv.")

st.markdown("---")

# -------------------------------- NAVIGATION ------------------------------
tabs = st.tabs(["📈  Glucose history", "🎯  Forecast evaluation", "⚡  Event evaluation", "▦  Data explorer", "📊  Model summary", "ⓘ  About"])

# -------------------------------- HISTORY ---------------------------------
with tabs[0]:
    section("Glucose timeline", "Inspect simulated glucose observations and optional activity or heart-rate signals.")
    if patient_history.empty:
        st.info("No time-series records match this patient. Verify that synthetic_timeseries.csv contains patient IDs that match the selector.")
        if history_error:
            st.caption(history_error)
        elif not history.empty:
            st.caption("Available time-series IDs: " + ", ".join(clean_ids(history)))
            with st.expander("Inspect source columns"):
                st.write(history.columns.tolist())
    else:
        glucose_col = find_col(patient_history, ["glucose_mg_dl", "glucose", "blood_glucose", "glucose_value", "glucose_level"])
        if not glucose_col:
            st.error("A recognizable glucose column was not found in the time-series data.")
            st.caption("Expected a field such as glucose_mg_dl. Available fields: " + ", ".join(patient_history.columns))
            show_table(patient_history.tail(100), f"{selected_patient}_history")
        else:
            patient_history[glucose_col] = numeric(patient_history[glucose_col])
            usable = patient_history.dropna(subset=[glucose_col]).copy()
            if usable.empty:
                st.warning("The glucose column exists but has no valid numeric observations.")
                show_table(patient_history.tail(100), f"{selected_patient}_history")
            else:
                window = st.select_slider("Observation window", options=[24, 48, 96, 168, 336, 672], value=168, format_func=lambda n: f"Last {n} records")
                recent = usable.tail(window)
                xcol = timestamp_x(recent)
                x = recent[xcol] if xcol else np.arange(1, len(recent) + 1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=recent[glucose_col], mode="lines", name="Synthetic glucose", line=dict(color="#087f8c", width=2.8), fill="tozeroy", fillcolor="rgba(8,127,140,.08)", hovertemplate="%{y:.1f} mg/dL<extra></extra>"))
                fig.add_hrect(y0=70, y1=180, fillcolor="rgba(8,127,140,.055)", line_width=0, layer="below", annotation_text="Illustrative reference band", annotation_position="top left")
                plot_style(fig, "Glucose (mg/dL)", xcol or "Record number")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)
                latest = recent.iloc[-1]
                hr_col = find_col(recent, ["heart_rate_bpm", "heart_rate", "bpm"])
                steps_col = find_col(recent, ["steps_30min", "steps", "step_count"])
                a, b, c = st.columns(3)
                a.metric("Latest synthetic glucose", f"{latest[glucose_col]:.1f} mg/dL")
                hr = safe_number(latest[hr_col]) if hr_col else None
                steps = safe_number(latest[steps_col]) if steps_col else None
                b.metric("Heart rate", f"{hr:.0f} bpm" if hr is not None else "Not available")
                c.metric("Steps", f"{steps:,.0f}" if steps is not None else "Not available")
                with st.expander("Inspect recent records"):
                    show_table(recent.tail(150), f"{selected_patient}_recent_history")
    st.caption("The shaded band is illustrative only and is not a personalized clinical target.")

# ----------------------------- FORECAST EVALUATION ------------------------
with tabs[1]:
    section("Forecast model performance", "Compare retrospective model predictions with observed synthetic values and a baseline where available.")
    if patient_forecast.empty:
        st.info("No forecast evaluation rows match this patient.")
        if forecast_error:
            st.caption(forecast_error)
        elif not forecast.empty:
            st.caption("Available forecast IDs: " + ", ".join(clean_ids(forecast)))
    else:
        actual_col = find_col(patient_forecast, ["actual_glucose_mg_dl", "actual_glucose", "actual", "y_true", "actual_value"])
        pred_col = find_col(patient_forecast, ["model_prediction_mg_dl", "predicted_glucose_mg_dl", "model_prediction", "prediction", "predicted", "y_pred"])
        base_col = find_col(patient_forecast, ["persistence_prediction_mg_dl", "baseline_glucose_mg_dl", "baseline_prediction", "persistence_prediction", "baseline"])
        if not actual_col or not pred_col:
            st.error("Forecast metrics require recognizable actual-value and model-prediction columns.")
            st.caption("Detected columns: " + ", ".join(patient_forecast.columns))
            show_table(patient_forecast, f"{selected_patient}_forecast")
        else:
            work = patient_forecast.copy()
            work[actual_col] = numeric(work[actual_col])
            work[pred_col] = numeric(work[pred_col])
            needed = [actual_col, pred_col]
            if base_col:
                work[base_col] = numeric(work[base_col])
            work = work.dropna(subset=needed)
            if work.empty:
                st.warning("No complete numeric rows are available to calculate forecast metrics.")
            else:
                actual = work[actual_col].to_numpy(dtype=float)
                pred = work[pred_col].to_numpy(dtype=float)
                model_mae = mean_absolute_error(actual, pred)
                model_rmse = float(np.sqrt(mean_squared_error(actual, pred)))
                if base_col:
                    base_rows = work.dropna(subset=[base_col])
                    if not base_rows.empty:
                        baseline_mae = mean_absolute_error(base_rows[actual_col], base_rows[base_col])
                        baseline_rmse = float(np.sqrt(mean_squared_error(base_rows[actual_col], base_rows[base_col])))
                        a, b, c, d = st.columns(4)
                        a.metric("Model MAE", f"{model_mae:.2f} mg/dL")
                        b.metric("Model RMSE", f"{model_rmse:.2f} mg/dL")
                        c.metric("Baseline MAE", f"{baseline_mae:.2f} mg/dL")
                        d.metric("Baseline RMSE", f"{baseline_rmse:.2f} mg/dL")
                        if baseline_mae > 0:
                            st.caption(f"Model MAE change versus baseline: {((baseline_mae-model_mae)/baseline_mae)*100:+.1f}% (positive means lower model error).")
                    else:
                        st.metric("Model MAE", f"{model_mae:.2f} mg/dL")
                        st.metric("Model RMSE", f"{model_rmse:.2f} mg/dL")
                        st.info("Baseline column exists, but has no usable numeric rows.")
                else:
                    a, b = st.columns(2)
                    a.metric("Model MAE", f"{model_mae:.2f} mg/dL")
                    b.metric("Model RMSE", f"{model_rmse:.2f} mg/dL")
                    st.info("No baseline prediction column was found; baseline metrics are omitted.")
                xcol = timestamp_x(work)
                x = work[xcol] if xcol else np.arange(1, len(work)+1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=work[actual_col], name="Actual", mode="lines", line=dict(color="#172b40", width=2.5)))
                fig.add_trace(go.Scatter(x=x, y=work[pred_col], name="Model prediction", mode="lines", line=dict(color="#087f8c", width=2.2)))
                if base_col and work[base_col].notna().any():
                    fig.add_trace(go.Scatter(x=x, y=work[base_col], name="Baseline", mode="lines", line=dict(color="#9277b5", width=2, dash="dash")))
                plot_style(fig, "Glucose (mg/dL)", xcol or "Evaluation row")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("View evaluation records"):
                    show_table(work, f"{selected_patient}_forecast")
                st.download_button("Download forecast evaluation", work.to_csv(index=False).encode("utf-8"), f"{selected_patient}_forecast_evaluation.csv", "text/csv", key="forecast_download")
    st.caption("These are retrospective results on synthetic data, not live forecasts or evidence of clinical performance.")

# ------------------------------- EVENT EVALUATION -------------------------
with tabs[2]:
    section("Event-classification evaluation", "Review binary event predictions and common classification metrics when valid labels are available.")
    if patient_events.empty:
        st.info("No event evaluation rows match this patient.")
        if events_error:
            st.caption(events_error)
        elif not events.empty:
            st.caption("Available event IDs: " + ", ".join(clean_ids(events)))
    else:
        actual_col = find_col(patient_events, ["actual_event", "actual", "y_true", "true_event", "actual_label", "event_true"])
        pred_col = find_col(patient_events, ["predicted_event", "prediction", "predicted", "y_pred", "predicted_label", "event_pred"])
        prob_col = find_col(patient_events, ["predicted_probability", "event_probability", "probability", "predicted_proba", "event_score"])
        if not actual_col or not pred_col:
            st.error("Event evaluation requires recognizable actual-label and predicted-label columns.")
            st.caption("Detected columns: " + ", ".join(patient_events.columns))
            show_table(patient_events, f"{selected_patient}_events")
        else:
            valid = patient_events.copy()
            valid["_actual_binary"] = binary_values(valid[actual_col])
            valid["_pred_binary"] = binary_values(valid[pred_col])
            valid = valid.dropna(subset=["_actual_binary", "_pred_binary"])
            if valid.empty:
                st.warning("No valid binary event labels were found. Supported values include 0/1, true/false, yes/no, positive/negative, and event/no_event.")
                show_table(patient_events, f"{selected_patient}_events")
            else:
                actual = valid["_actual_binary"].astype(int)
                pred = valid["_pred_binary"].astype(int)
                tp = int(((actual == 1) & (pred == 1)).sum())
                tn = int(((actual == 0) & (pred == 0)).sum())
                fp = int(((actual == 0) & (pred == 1)).sum())
                fn = int(((actual == 1) & (pred == 0)).sum())
                precision = tp/(tp+fp) if tp+fp else 0.0
                recall = tp/(tp+fn) if tp+fn else 0.0
                specificity = tn/(tn+fp) if tn+fp else 0.0
                accuracy = (tp+tn)/len(valid)
                f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
                cols = st.columns(5)
                for col, label, value in zip(cols, ["Accuracy", "Precision", "Recall", "Specificity", "F1 score"], [f"{accuracy:.1%}", f"{precision:.1%}", f"{recall:.1%}", f"{specificity:.1%}", f"{f1:.3f}"]):
                    col.metric(label, value)
                left, right = st.columns([1, 1.25])
                with left:
                    st.markdown("#### Confusion matrix")
                    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: no event", "Actual: event"], columns=["Predicted: no event", "Predicted: event"])
                    st.dataframe(cm, use_container_width=True)
                    st.caption(f"TN {tn} · FP {fp} · FN {fn} · TP {tp}")
                with right:
                    if prob_col:
                        valid[prob_col] = numeric(valid[prob_col])
                        prob_rows = valid.dropna(subset=[prob_col])
                        if not prob_rows.empty:
                            xcol = timestamp_x(prob_rows)
                            x = prob_rows[xcol] if xcol else np.arange(1, len(prob_rows)+1)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=x, y=prob_rows[prob_col], name="Event score", mode="lines", line=dict(color="#087f8c", width=2.5)))
                            fig.add_hline(y=.5, line_dash="dash", line_color="#9277b5", annotation_text="0.5 reference")
                            plot_style(fig, "Model score", xcol or "Evaluation row")
                            fig.update_yaxes(range=[0, 1])
                            fig.update_layout(height=330)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("The event-score column has no numeric values.")
                    else:
                        st.info("No probability/score column is available for a score timeline.")
                with st.expander("View event evaluation records"):
                    show_table(valid.drop(columns=["_actual_binary", "_pred_binary"], errors="ignore"), f"{selected_patient}_events")
                st.download_button("Download event evaluation", valid.drop(columns=["_actual_binary", "_pred_binary"], errors="ignore").to_csv(index=False).encode("utf-8"), f"{selected_patient}_event_evaluation.csv", "text/csv", key="event_download")
    st.warning("Event metrics are exploratory synthetic results. This classifier is not a medical alert system and must not guide treatment decisions.")

# -------------------------------- DATA EXPLORER ----------------------------
with tabs[3]:
    section("Data explorer", "Inspect source datasets, filter by the selected profile, and export records for reproducibility.")
    datasets = {"Patient profiles": (ehr, ehr_error), "Glucose time series": (history, history_error), "Forecast evaluation": (forecast, forecast_error), "Event evaluation": (events, events_error)}
    selected_dataset = st.selectbox("Dataset", list(datasets.keys()), key="explorer_dataset")
    source_df, source_error = datasets[selected_dataset]
    if source_df.empty:
        st.info(f"No data is available for {selected_dataset.lower()}.")
        if source_error:
            st.caption(source_error)
    else:
        f1, f2 = st.columns([1, 1])
        with f1:
            filter_patient = st.toggle("Filter to selected patient", value=True, key="explorer_patient_filter")
        with f2:
            search_text = st.text_input("Search visible values", placeholder="Optional keyword…", key="explorer_search")
        visible = subset_patient(source_df, selected_patient) if filter_patient and selected_patient and find_col(source_df, ["patient_id", "patient", "subject_id", "subject"]) else source_df.copy()
        if search_text.strip() and not visible.empty:
            mask = visible.astype(str).apply(lambda col: col.str.contains(search_text.strip(), case=False, na=False, regex=False)).any(axis=1)
            visible = visible.loc[mask]
        if visible.empty:
            st.info("No rows match the current filters.")
        else:
            st.caption(f"{selected_dataset} · {len(visible):,} rows · {len(visible.columns):,} columns")
            st.dataframe(pretty_df(visible), hide_index=True, use_container_width=True, height=430)
            st.download_button("Download filtered dataset", visible.to_csv(index=False).encode("utf-8"), f"{selected_dataset.lower().replace(' ','_')}.csv", "text/csv", key="explorer_download")
            with st.expander("Technical field reference"):
                ref = pd.DataFrame({"Display name": [str(c).replace("_", " ").title() for c in visible.columns], "Original column": list(visible.columns), "Data type": [str(visible[c].dtype) for c in visible.columns], "Non-null values": [int(visible[c].notna().sum()) for c in visible.columns]})
                st.dataframe(ref, hide_index=True, use_container_width=True)

# -------------------------------- MODEL SUMMARY ----------------------------
with tabs[4]:
    section("Model summary", "A cross-patient summary of the available retrospective evaluation outputs.")
    if not forecast.empty:
        actual_col = find_col(forecast, ["actual_glucose_mg_dl", "actual_glucose", "actual", "y_true", "actual_value"])
        pred_col = find_col(forecast, ["model_prediction_mg_dl", "predicted_glucose_mg_dl", "model_prediction", "prediction", "predicted", "y_pred"])
        base_col = find_col(forecast, ["persistence_prediction_mg_dl", "baseline_glucose_mg_dl", "baseline_prediction", "persistence_prediction", "baseline"])
        if actual_col and pred_col:
            f = forecast.copy()
            f[actual_col] = numeric(f[actual_col]); f[pred_col] = numeric(f[pred_col])
            f = f.dropna(subset=[actual_col, pred_col])
            if not f.empty:
                actual = f[actual_col].to_numpy(dtype=float); pred = f[pred_col].to_numpy(dtype=float)
                mae = mean_absolute_error(actual, pred); rmse = float(np.sqrt(mean_squared_error(actual, pred)))
                m1, m2, m3 = st.columns(3)
                m1.metric("Forecast MAE", f"{mae:.2f} mg/dL")
                m2.metric("Forecast RMSE", f"{rmse:.2f} mg/dL")
                m3.metric("Evaluation rows", f"{len(f):,}")
                if base_col:
                    f[base_col] = numeric(f[base_col]); fb = f.dropna(subset=[base_col])
                    if not fb.empty:
                        baseline_mae = mean_absolute_error(fb[actual_col], fb[base_col])
                        a, b = st.columns(2)
                        a.metric("Baseline MAE", f"{baseline_mae:.2f} mg/dL")
                        if baseline_mae > 0:
                            change = (baseline_mae-mae)/baseline_mae*100
                            b.metric("MAE change vs baseline", f"{change:+.1f}%", help="Positive means the model has lower MAE than the baseline.")
                else:
                    st.info("No recognized baseline column was found; baseline comparison is omitted.")
            else:
                st.info("Forecast data loaded, but no complete numeric rows are available.")
        else:
            st.info("Forecast file loaded, but recognizable actual/prediction columns were not found.")
            st.caption("Detected columns: " + ", ".join(forecast.columns))
    else:
        st.info(forecast_error or "Forecast evaluation data is empty.")

    st.markdown("### Event classification")
    if not events.empty:
        a_col = find_col(events, ["actual_event", "actual", "y_true", "true_event", "actual_label", "event_true"])
        p_col = find_col(events, ["predicted_event", "prediction", "predicted", "y_pred", "predicted_label", "event_pred"])
        if a_col and p_col:
            ev = events.copy(); ev["_a"] = binary_values(ev[a_col]); ev["_p"] = binary_values(ev[p_col]); ev = ev.dropna(subset=["_a", "_p"])
            if not ev.empty:
                actual = ev["_a"].astype(int); pred = ev["_p"].astype(int)
                tp = int(((actual==1)&(pred==1)).sum()); tn = int(((actual==0)&(pred==0)).sum()); fp = int(((actual==0)&(pred==1)).sum()); fn = int(((actual==1)&(pred==0)).sum())
                precision = tp/(tp+fp) if tp+fp else 0.; recall = tp/(tp+fn) if tp+fn else 0.; specificity = tn/(tn+fp) if tn+fp else 0.; accuracy = (tp+tn)/len(ev); f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
                a,b,c,d,e = st.columns(5)
                a.metric("Accuracy", f"{accuracy:.1%}"); b.metric("Precision", f"{precision:.1%}"); c.metric("Recall", f"{recall:.1%}"); d.metric("Specificity", f"{specificity:.1%}"); e.metric("F1 score", f"{f1:.3f}")
            else:
                st.info("No valid binary event rows are available for summary metrics.")
        else:
            st.info("Event file loaded, but recognizable actual/predicted columns were not found.")
    else:
        st.info(events_error or "Event evaluation data is empty.")
    st.markdown("### How to interpret these metrics")
    st.write("Lower MAE and RMSE indicate smaller retrospective forecast error. Precision and recall capture different aspects of event classification. These metrics depend on the synthetic dataset and do not establish real-world or clinical performance.")

# -------------------------------- ABOUT ------------------------------------
with tabs[5]:
    section("About GlucoTwin", "Project scope, reproducibility, data availability, and responsible-use notes.")
    st.markdown("### Project purpose")
    st.write("GlucoTwin is a research prototype for exploring synthetic patient profiles, simulated glucose time series, retrospective forecasting evaluation, and binary event-classification experiments.")
    st.markdown("### Dataset health")
    for label, frame, err in [("Patient profiles",ehr,ehr_error),("Glucose time series",history,history_error),("Forecast evaluation",forecast,forecast_error),("Event evaluation",events,events_error)]:
        status_card(label, frame, err)
    st.markdown("### Project structure")
    st.code("GlucoTwin/\n├── app.py\n├── requirements.txt\n└── data/\n    ├── synthetic_ehr.csv\n    ├── synthetic_timeseries.csv\n    ├── forecast_evaluation.csv\n    └── event_evaluation.csv", language="text")
    st.markdown("### Limitations and responsible use")
    st.write("All records in this demo are synthetic. Retrospective metrics do not establish clinical validity. The app is not a medical device and must not be used for diagnosis, treatment decisions, or real-world medical alerts.")

# -------------------------------- FOOTER -----------------------------------
st.markdown("""
<div class="gt-footer"><b>GlucoTwin · Research Intelligence</b><br>Digital health research prototype · Synthetic data only<br>Not for diagnosis, treatment decisions, or real-world medical alerts.</div>
""", unsafe_allow_html=True)