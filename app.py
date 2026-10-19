from pathlib import Path
import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# GlucoTwin | Research Intelligence Dashboard  (v2 redesign)
# Defensive UI for synthetic research data; not a medical tool.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

C = dict(
    ink="#0f1b2d", navy="#12263f", teal="#0e9aa7", teal_soft="rgba(14,154,167,.10)",
    indigo="#5b6cff", amber="#f59e0b", rose="#e5484d", green="#12a150",
    slate="#64748b", grid="#eef2f7", line="#e5eaf1", canvas="#f6f8fb",
)

st.set_page_config(
    page_title="GlucoTwin | Research Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ DESIGN SYSTEM ------------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{
  --ink:#0f1b2d; --navy:#12263f; --teal:#0e9aa7; --indigo:#5b6cff; --amber:#f59e0b;
  --rose:#e5484d; --green:#12a150; --muted:#64748b; --line:#e5eaf1; --canvas:#f6f8fb;
  --radius:16px; --shadow:0 1px 2px rgba(15,27,45,.04),0 8px 24px rgba(15,27,45,.05);
}
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:var(--canvas);color:var(--ink);}
.block-container{max-width:1480px;padding-top:1.2rem;padding-bottom:3rem;}
header[data-testid="stHeader"]{background:transparent;}
h1,h2,h3,h4{font-family:'Manrope',sans-serif!important;color:var(--ink)!important;letter-spacing:-.02em;}
h4{font-size:1rem!important;font-weight:700!important;margin-top:.2rem;}
p,li{line-height:1.65;}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(170deg,#0f1b2d 0%,#14344d 65%,#0b5560 130%);border-right:1px solid #223c52;}
[data-testid="stSidebar"] *{color:#e8f0f7;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] small{color:#a9bccb!important;}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#fff!important;border:1px solid #d5e0e9!important;border-radius:12px!important;min-height:46px;}
[data-testid="stSidebar"] [data-baseweb="select"],[data-testid="stSidebar"] [data-baseweb="select"] *{color:#0f1b2d!important;-webkit-text-fill-color:#0f1b2d!important;opacity:1!important;}
[data-testid="stSidebar"] [data-baseweb="select"] input{caret-color:#0f1b2d;}
[data-testid="stSidebar"] [data-baseweb="select"] svg{fill:#0f1b2d!important;}
[data-testid="stSidebar"] hr{border-color:#ffffff24!important;}
/* Sidebar expander: transparent so light text stays readable on the dark sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"],[data-testid="stSidebar"] [data-testid="stExpander"] details,[data-testid="stSidebar"] [data-testid="stExpander"] summary{background:transparent!important;}
[data-testid="stSidebar"] [data-testid="stExpander"]{border:1px solid #ffffff2e;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover{background:#ffffff12!important;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary,[data-testid="stSidebar"] [data-testid="stExpander"] summary *{color:#e8f0f7!important;-webkit-text-fill-color:#e8f0f7!important;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg{fill:#e8f0f7!important;stroke:#e8f0f7!important;}
.gt-brand{display:flex;align-items:center;gap:10px;font-family:'Manrope',sans-serif;font-weight:800;font-size:22px;letter-spacing:-.6px;}
.gt-logo{width:36px;height:36px;border-radius:11px;display:grid;place-items:center;background:linear-gradient(135deg,#0e9aa7,#5b6cff);font-size:18px;}
.gt-brand-sub{font-size:10px;text-transform:uppercase;letter-spacing:1.8px;color:#a9bccb;margin:4px 0 0 46px;}
.gt-side-label{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;color:#8fa6b8;margin:6px 0 6px;}

/* Hero */
.gt-hero{position:relative;overflow:hidden;background:linear-gradient(120deg,#0f1b2d 0%,#14344d 58%,#0e9aa7 140%);border-radius:22px;padding:30px 34px;color:#fff;margin-bottom:14px;box-shadow:0 18px 40px rgba(15,27,45,.14);}
.gt-hero:before{content:'';position:absolute;right:-70px;top:-110px;width:300px;height:300px;border-radius:50%;border:1px solid #ffffff1c;box-shadow:0 0 0 34px #ffffff08,0 0 0 70px #ffffff05;}
.gt-hero *{color:#fff!important;position:relative;z-index:1;}
.gt-eyebrow{font-size:10px;letter-spacing:2px;text-transform:uppercase;font-weight:700;opacity:.75;}
.gt-title{font-family:'Manrope',sans-serif;font-size:clamp(26px,3vw,40px);font-weight:800;letter-spacing:-1.3px;line-height:1.15;margin-top:8px;}
.gt-subtitle{font-size:14px;opacity:.85;margin-top:10px;line-height:1.7;max-width:760px;}
.gt-hero-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px;}
.gt-hchip{border:1px solid #ffffff3d;background:#ffffff14;padding:6px 12px;border-radius:999px;font-size:11px;font-weight:600;letter-spacing:.3px;}

/* Notice */
.gt-notice{display:flex;gap:12px;align-items:flex-start;background:#fffbeb;border:1px solid #fde8a8;color:#7a5200;border-radius:12px;padding:11px 15px;font-size:12.5px;line-height:1.6;margin-bottom:14px;}
.gt-notice b{color:#7a5200;}

/* Section headings */
.gt-section{font-family:'Manrope',sans-serif;font-size:20px;font-weight:800;color:var(--ink);letter-spacing:-.5px;margin:6px 0 2px;}
.gt-description{font-size:13px;color:var(--muted);margin-bottom:14px;line-height:1.6;}

/* Cards */
.gt-card{background:#fff;border:1px solid var(--line);border-radius:var(--radius);padding:18px 20px;box-shadow:var(--shadow);}
.kpi{background:#fff;border:1px solid var(--line);border-radius:var(--radius);padding:15px 17px;box-shadow:var(--shadow);height:100%;position:relative;overflow:hidden;}
.kpi:before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--line);}
.kpi-teal:before{background:var(--teal);} .kpi-indigo:before{background:var(--indigo);}
.kpi-amber:before{background:var(--amber);} .kpi-rose:before{background:var(--rose);} .kpi-green:before{background:var(--green);}
.kpi-label{font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:700;color:#7a8ca0;}
.kpi-value{font-family:'Manrope',sans-serif;font-size:26px;font-weight:800;letter-spacing:-.8px;color:var(--ink);margin-top:4px;line-height:1.2;}
.kpi-note{font-size:12px;color:var(--muted);margin-top:3px;min-height:16px;}
.kpi-up{color:var(--green);font-weight:600;} .kpi-down{color:var(--rose);font-weight:600;}

/* Profile card */
.gt-profile{display:flex;gap:16px;align-items:center;background:#fff;border:1px solid var(--line);border-radius:var(--radius);padding:16px 20px;box-shadow:var(--shadow);}
.gt-avatar{width:56px;height:56px;border-radius:16px;display:grid;place-items:center;font-family:'Manrope',sans-serif;font-weight:800;font-size:20px;color:#fff;background:linear-gradient(135deg,#0e9aa7,#5b6cff);flex-shrink:0;}
.gt-pname{font-family:'Manrope',sans-serif;font-weight:800;font-size:18px;letter-spacing:-.4px;}
.gt-psub{font-size:12px;color:var(--muted);margin-top:2px;}
.gt-chip{display:inline-block;padding:4px 10px;border-radius:999px;background:#eef6f8;color:#0b7a85;font-size:11.5px;font-weight:600;margin:6px 6px 0 0;}
.gt-chip-muted{background:#f1f4f8;color:#556579;}

/* Status */
.gt-status{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:9px 12px;background:#ffffff0f;border:1px solid #ffffff1f;border-radius:10px;margin:5px 0;font-size:12px;}
.gt-status-main{background:#fff;border:1px solid var(--line);color:var(--ink);}
.gt-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;}

/* Empty state */
.gt-empty{text-align:center;background:#fff;border:1px dashed #cfd9e5;border-radius:var(--radius);padding:34px 20px;color:var(--muted);}
.gt-empty .ico{font-size:30px;margin-bottom:6px;} .gt-empty b{color:var(--ink);display:block;font-size:15px;margin-bottom:4px;}

/* Native widgets */
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:var(--radius);padding:8px;box-shadow:var(--shadow);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#fff;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);}
[data-testid="stTabs"] button[data-baseweb="tab"]{padding:11px 16px;border-radius:10px 10px 0 0;font-weight:600;color:#556579;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--teal)!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:var(--teal)!important;height:3px;}
.stButton>button,.stDownloadButton>button{border-radius:10px;min-height:40px;font-weight:600;border:1px solid #d9e2eb;background:#fff;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--teal);color:var(--teal);}
div[role="radiogroup"]{gap:6px;}
.gt-footer{margin-top:34px;padding:18px 0;border-top:1px solid #dce5ee;color:#64748b;font-size:12px;line-height:1.8;text-align:center;}
@media(max-width:768px){.block-container{padding-left:1rem;padding-right:1rem}.gt-hero{padding:22px}.kpi-value{font-size:22px}.gt-profile{flex-direction:column;align-items:flex-start}}
</style>
""", unsafe_allow_html=True)


# -------------------------------- HELPERS ----------------------------------
def esc(value):
    return html.escape(str(value))


def find_col(df, candidates):
    if df is None or df.empty:
        return None
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for name in candidates:
        key = re.sub(r"[^a-z0-9]", "", str(name).lower())
        if key in lookup:
            return lookup[key]
    return None


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


ID_NAMES = ["patient_id", "patient", "subject_id", "subject"]


def clean_ids(df):
    col = find_col(df, ID_NAMES)
    if df is None or df.empty or not col:
        return []
    vals = df[col].dropna().astype(str).str.strip()
    return sorted(v for v in vals.unique().tolist() if v and v.lower() not in {"nan", "none", "null"})


def subset_patient(df, patient_id):
    if df is None or df.empty or not patient_id:
        return pd.DataFrame() if df is None else df.iloc[0:0].copy()
    col = find_col(df, ID_NAMES)
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


# ------------------------------ UI COMPONENTS ------------------------------
def section(title, description=""):
    st.markdown(
        f'<div class="gt-section">{esc(title)}</div><div class="gt-description">{esc(description)}</div>',
        unsafe_allow_html=True,
    )


def kpi(label, value, note="", tone="default", note_class=""):
    st.markdown(
        f'<div class="kpi kpi-{tone}"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-note {note_class}">{esc(note)}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of (label, value, note, tone, note_class)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            kpi(*item)


def empty_state(title, message, icon="🗂️"):
    st.markdown(
        f'<div class="gt-empty"><div class="ico">{icon}</div><b>{esc(title)}</b>{esc(message)}</div>',
        unsafe_allow_html=True,
    )


def notice(text):
    st.markdown(f'<div class="gt-notice"><div>⚠️</div><div>{text}</div></div>', unsafe_allow_html=True)


def show_table(df, key, height=340):
    if df is None or df.empty:
        empty_state("No records", "There are no records to display for this selection.")
        return
    st.caption(f"{len(df):,} records · {len(df.columns):,} fields")
    st.dataframe(pretty_df(df), hide_index=True, use_container_width=True, height=height)
    st.download_button(
        "⬇ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{key}.csv", mime="text/csv", key=f"download_{key}",
    )


def status_row(label, df, error, dark=True):
    ok = error is None and not df.empty
    color = "#34d399" if ok else "#fbbf24"
    detail = f"{len(df):,} rows" if ok else "Needs attention"
    cls = "gt-status" if dark else "gt-status gt-status-main"
    tip = "" if ok else f' title="{esc(error or "No rows available")}"'
    st.markdown(
        f'<div class="{cls}"{tip}><span><span class="gt-dot" style="background:{color}"></span>{esc(label)}</span>'
        f'<span style="opacity:.75">{esc(detail)}</span></div>',
        unsafe_allow_html=True,
    )


def plot_style(fig, y_title="", x_title="", height=400):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=height,
        font=dict(family="Inter, sans-serif", color="#34465b", size=12),
        margin=dict(l=12, r=12, t=30, b=12), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_yaxes(title_text=y_title, gridcolor=C["grid"], zerolinecolor=C["grid"])
    fig.update_xaxes(title_text=x_title, gridcolor=C["grid"], showspikes=True, spikecolor="#cbd5e1", spikethickness=1)
    return fig


# ------------------------------ METRIC LOGIC -------------------------------
FORECAST_ACTUAL = ["actual_glucose_mg_dl", "actual_glucose", "actual", "y_true", "actual_value"]
FORECAST_PRED = ["model_prediction_mg_dl", "predicted_glucose_mg_dl", "model_prediction", "prediction", "predicted", "y_pred"]
FORECAST_BASE = ["persistence_prediction_mg_dl", "baseline_glucose_mg_dl", "baseline_prediction", "persistence_prediction", "baseline"]
EVENT_ACTUAL = ["actual_event", "actual", "y_true", "true_event", "actual_label", "event_true"]
EVENT_PRED = ["predicted_event", "prediction", "predicted", "y_pred", "predicted_label", "event_pred"]
EVENT_PROB = ["predicted_probability", "event_probability", "probability", "predicted_proba", "event_score"]


def rmse_of(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def forecast_prepare(df):
    """Return (clean_df, actual_col, pred_col, base_col) or (None, a, p, b) if columns missing."""
    a = find_col(df, FORECAST_ACTUAL)
    p = find_col(df, FORECAST_PRED)
    b = find_col(df, FORECAST_BASE)
    if not a or not p:
        return None, a, p, b
    work = df.copy()
    work[a] = numeric(work[a])
    work[p] = numeric(work[p])
    if b:
        work[b] = numeric(work[b])
    work = work.dropna(subset=[a, p])
    return work, a, p, b


def forecast_stats(work, a, p, b):
    stats = {"mae": mean_absolute_error(work[a], work[p]), "rmse": rmse_of(work[a], work[p]), "n": len(work)}
    if b:
        rows = work.dropna(subset=[b])
        if not rows.empty:
            stats["b_mae"] = mean_absolute_error(rows[a], rows[b])
            stats["b_rmse"] = rmse_of(rows[a], rows[b])
            stats["change"] = (stats["b_mae"] - stats["mae"]) / stats["b_mae"] * 100 if stats["b_mae"] > 0 else None
    return stats


def event_prepare(df):
    a = find_col(df, EVENT_ACTUAL)
    p = find_col(df, EVENT_PRED)
    pr = find_col(df, EVENT_PROB)
    if not a or not p:
        return None, a, p, pr
    valid = df.copy()
    valid["_a"] = binary_values(valid[a])
    valid["_p"] = binary_values(valid[p])
    valid = valid.dropna(subset=["_a", "_p"])
    return valid, a, p, pr


def event_stats(valid):
    actual = valid["_a"].astype(int)
    pred = valid["_p"].astype(int)
    tp = int(((actual == 1) & (pred == 1)).sum())
    tn = int(((actual == 0) & (pred == 0)).sum())
    fp = int(((actual == 0) & (pred == 1)).sum())
    fn = int(((actual == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall,
        specificity=tn / (tn + fp) if tn + fp else 0.0,
        accuracy=(tp + tn) / len(valid) if len(valid) else 0.0,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        prevalence=(tp + fn) / len(valid) if len(valid) else 0.0, n=len(valid),
    )


def event_kpis(s):
    kpi_row([
        ("Accuracy", f"{s['accuracy']:.1%}", f"{s['n']:,} labelled rows", "teal", ""),
        ("Precision", f"{s['precision']:.1%}", "Of predicted events, share correct", "indigo", ""),
        ("Recall", f"{s['recall']:.1%}", "Of real events, share caught", "indigo", ""),
        ("Specificity", f"{s['specificity']:.1%}", "Of non-events, share cleared", "teal", ""),
        ("F1 score", f"{s['f1']:.3f}", f"Event prevalence {s['prevalence']:.1%}", "amber", ""),
    ])


# -------------------------------- LOAD DATA --------------------------------
ehr, ehr_error = load_csv("synthetic_ehr.csv")
history, history_error = load_csv("synthetic_timeseries.csv", ["timestamp", "datetime", "date"])
forecast, forecast_error = load_csv("forecast_evaluation.csv", ["timestamp", "datetime", "date"])
events, events_error = load_csv("event_evaluation.csv", ["timestamp", "datetime", "date"])
SOURCES = [
    ("Patient profiles", ehr, ehr_error),
    ("Glucose time series", history, history_error),
    ("Forecast evaluation", forecast, forecast_error),
    ("Event evaluation", events, events_error),
]

# -------------------------------- SIDEBAR ----------------------------------
with st.sidebar:
    st.markdown(
        '<div class="gt-brand"><div class="gt-logo">🧬</div>GlucoTwin</div>'
        '<div class="gt-brand-sub">Research intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="gt-side-label">Workspace</div>', unsafe_allow_html=True)

    all_ids = sorted(set(clean_ids(ehr) + clean_ids(history) + clean_ids(forecast) + clean_ids(events)))
    if all_ids:
        current = st.session_state.get("gt_patient", all_ids[0])
        if current not in all_ids:
            current = all_ids[0]
        selected_patient = st.selectbox(
            "Synthetic patient", options=all_ids, index=all_ids.index(current), key="gt_patient",
            help="Filters every patient-level view where patient IDs are available.",
        )
        idx = all_ids.index(selected_patient)
        st.caption(f"Profile {idx + 1} of {len(all_ids)}")
    else:
        selected_patient = None
        st.error("No patient IDs were found in the available CSV files.")

    st.markdown("<hr>", unsafe_allow_html=True)
    ready = sum(1 for _, d, e in SOURCES if e is None and not d.empty)
    with st.expander(f"Data sources · {ready}/{len(SOURCES)} ready", expanded=ready < len(SOURCES)):
        for label, d, e in SOURCES:
            status_row(label, d, e)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Synthetic data only · Research prototype")

# -------------------------------- HEADER -----------------------------------
st.markdown(
    f"""
<div class="gt-hero">
  <div class="gt-eyebrow">Digital health · Research workspace</div>
  <div class="gt-title">GlucoTwin Intelligence</div>
  <div class="gt-subtitle">Explore simulated glucose trends, review retrospective forecast performance, and inspect event-classification experiments, all in one workspace.</div>
  <div class="gt-hero-chips">
    <span class="gt-hchip">● Synthetic research environment</span>
    <span class="gt-hchip">Viewing: {esc(selected_patient) if selected_patient else "no profile"}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
notice("<b>Research demonstration only.</b> All records and results are synthetic. This application is not clinically validated and must not be used for diagnosis, treatment, real-time alerts, or treatment decisions.")

if selected_patient:
    profile = subset_patient(ehr, selected_patient)
    patient_history = subset_patient(history, selected_patient)
    patient_forecast = subset_patient(forecast, selected_patient)
    patient_events = subset_patient(events, selected_patient)
else:
    profile = patient_history = patient_forecast = patient_events = pd.DataFrame()

for frame in (patient_history, patient_forecast, patient_events):
    if not frame.empty:
        dt = timestamp_x(frame)
        if dt:
            frame.sort_values(dt, inplace=True, na_position="last")

# ------------------------------ PATIENT OVERVIEW ---------------------------
section("Patient overview", "A snapshot of the selected synthetic profile and the records available for analysis.")
if not selected_patient:
    empty_state("No patient selected", "Check the files in the data/ folder and make sure they contain a patient_id column.", "🧪")
else:
    p = profile.iloc[0] if not profile.empty else pd.Series(dtype=object)
    age = safe_number(p.get(find_col(profile, ["age_years", "age"]))) if not profile.empty and find_col(profile, ["age_years", "age"]) else None
    dur_col = find_col(profile, ["diabetes_duration_years", "diabetes_duration"])
    a1c_col = find_col(profile, ["hba1c_percent", "hba1c", "hb_a1c"])
    duration = safe_number(p.get(dur_col)) if dur_col else None
    hba1c = safe_number(p.get(a1c_col)) if a1c_col else None

    initials = re.sub(r"[^A-Za-z0-9]", "", str(selected_patient))[-2:].upper() or "GT"
    chips = []
    chips.append(f'<span class="gt-chip">{age:.0f} years</span>' if age is not None else '<span class="gt-chip gt-chip-muted">Age n/a</span>')
    chips.append(f'<span class="gt-chip">{duration:.1f} yrs duration</span>' if duration is not None else '<span class="gt-chip gt-chip-muted">Duration n/a</span>')
    chips.append(f'<span class="gt-chip">HbA1c {hba1c:.1f}%</span>' if hba1c is not None else '<span class="gt-chip gt-chip-muted">HbA1c n/a</span>')
    left, right = st.columns([1.35, 2])
    with left:
        st.markdown(
            f'<div class="gt-profile"><div class="gt-avatar">{esc(initials)}</div><div>'
            f'<div class="gt-pname">{esc(selected_patient)}</div><div class="gt-psub">Synthetic profile</div>'
            f'{"".join(chips)}</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        kpi_row([
            ("Glucose observations", f"{len(patient_history):,}", "Time-series rows", "teal", ""),
            ("Forecast rows", f"{len(patient_forecast):,}", "Retrospective evaluation", "indigo", ""),
            ("Event rows", f"{len(patient_events):,}", "Classification evaluation", "amber", ""),
        ])
    if not profile.empty:
        with st.expander("View complete synthetic profile"):
            show_table(profile, f"{selected_patient}_profile", height=220)
    elif not ehr.empty:
        st.info(f"{selected_patient} appears in another dataset, but no matching profile row was found in synthetic_ehr.csv.")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# -------------------------------- NAVIGATION -------------------------------
tabs = st.tabs(["📈 Glucose history", "🎯 Forecast", "⚡ Events", "🔎 Data explorer", "📊 Model summary", "ⓘ About"])

# --------------------------------- HISTORY ---------------------------------
with tabs[0]:
    section("Glucose timeline", "Inspect simulated glucose observations with optional heart-rate or activity context.")
    if patient_history.empty:
        empty_state("No time-series records", "Verify that synthetic_timeseries.csv contains patient IDs matching the selector.", "📈")
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
                window = st.radio(
                    "Observation window", [24, 48, 96, 168, 336, 672], index=3, horizontal=True,
                    format_func=lambda n: f"Last {n}", key="history_window",
                )
                recent = usable.tail(window)
                xcol = timestamp_x(recent)
                x = recent[xcol] if xcol else np.arange(1, len(recent) + 1)
                g = recent[glucose_col]
                hr_col = find_col(recent, ["heart_rate_bpm", "heart_rate", "bpm"])
                steps_col = find_col(recent, ["steps_30min", "steps", "step_count"])
                ctx_col = hr_col or steps_col

                in_band = float(g.between(70, 180).mean() * 100)
                cv = float(g.std() / g.mean() * 100) if g.mean() else 0.0
                kpi_row([
                    ("Latest", f"{g.iloc[-1]:.0f} mg/dL", "Most recent synthetic value", "teal", ""),
                    ("Average", f"{g.mean():.0f} mg/dL", f"Range {g.min():.0f}–{g.max():.0f}", "indigo", ""),
                    ("In illustrative band", f"{in_band:.0f}%", "70–180 mg/dL, not a clinical target", "green", ""),
                    ("Variability (CV)", f"{cv:.0f}%", f"SD {g.std():.1f} mg/dL", "amber", ""),
                ])
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                rows = 2 if ctx_col else 1
                fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                    row_heights=[0.72, 0.28] if ctx_col else [1])
                fig.add_hrect(y0=70, y1=180, fillcolor=C["teal_soft"], line_width=0, layer="below", row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=x, y=g, mode="lines", name="Synthetic glucose",
                    line=dict(color=C["teal"], width=2.6, shape="spline", smoothing=.5),
                    fill="tozeroy", fillcolor="rgba(14,154,167,.07)", hovertemplate="%{y:.1f} mg/dL<extra></extra>",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=[x.iloc[-1] if hasattr(x, "iloc") else x[-1]], y=[g.iloc[-1]], mode="markers",
                    marker=dict(size=10, color=C["teal"], line=dict(color="white", width=2)),
                    name="Latest", hoverinfo="skip", showlegend=False,
                ), row=1, col=1)
                if ctx_col:
                    vals = numeric(recent[ctx_col])
                    if hr_col:
                        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines", name="Heart rate (bpm)",
                                                 line=dict(color=C["rose"], width=1.8)), row=2, col=1)
                        fig.update_yaxes(title_text="bpm", row=2, col=1)
                    else:
                        fig.add_trace(go.Bar(x=x, y=vals, name="Steps", marker_color=C["indigo"], opacity=.75), row=2, col=1)
                        fig.update_yaxes(title_text="steps", row=2, col=1)
                plot_style(fig, height=520 if ctx_col else 400)
                fig.update_yaxes(title_text="mg/dL", row=1, col=1)
                fig.add_annotation(text="Illustrative reference band", xref="paper", yref="paper", x=0.005, y=0.995,
                                   showarrow=False, font=dict(size=10, color=C["slate"]), xanchor="left", yanchor="top")
                st.plotly_chart(fig, use_container_width=True)

                if xcol and pd.api.types.is_datetime64_any_dtype(usable[xcol]) and usable[xcol].notna().sum() > 24:
                    with st.expander("Typical daily pattern (all records)"):
                        tmp = usable.dropna(subset=[xcol]).copy()
                        tmp["_hour"] = tmp[xcol].dt.hour
                        grp = tmp.groupby("_hour")[glucose_col]
                        mean, q1, q3 = grp.mean(), grp.quantile(.25), grp.quantile(.75)
                        pf = go.Figure()
                        pf.add_trace(go.Scatter(x=q3.index, y=q3.values, line=dict(width=0), showlegend=False, hoverinfo="skip"))
                        pf.add_trace(go.Scatter(x=q1.index, y=q1.values, line=dict(width=0), fill="tonexty",
                                                fillcolor="rgba(14,154,167,.15)", name="Interquartile range", hoverinfo="skip"))
                        pf.add_trace(go.Scatter(x=mean.index, y=mean.values, name="Hourly mean",
                                                line=dict(color=C["teal"], width=2.6), hovertemplate="%{y:.0f} mg/dL<extra></extra>"))
                        plot_style(pf, "mg/dL", "Hour of day", height=320)
                        st.plotly_chart(pf, use_container_width=True)
                with st.expander("Inspect recent records"):
                    show_table(recent.tail(150), f"{selected_patient}_recent_history")
    st.caption("The shaded band is illustrative only and is not a personalized clinical target.")

# ---------------------------- FORECAST EVALUATION --------------------------
with tabs[1]:
    section("Forecast model performance", "Compare retrospective predictions with observed synthetic values and a baseline.")
    if patient_forecast.empty:
        empty_state("No forecast rows", "No forecast evaluation rows match this patient.", "🎯")
        if forecast_error:
            st.caption(forecast_error)
        elif not forecast.empty:
            st.caption("Available forecast IDs: " + ", ".join(clean_ids(forecast)))
    else:
        work, a_col, p_col, b_col = forecast_prepare(patient_forecast)
        if work is None:
            st.error("Forecast metrics require recognizable actual-value and model-prediction columns.")
            st.caption("Detected columns: " + ", ".join(patient_forecast.columns))
            show_table(patient_forecast, f"{selected_patient}_forecast")
        elif work.empty:
            st.warning("No complete numeric rows are available to calculate forecast metrics.")
        else:
            s = forecast_stats(work, a_col, p_col, b_col)
            items = [
                ("Model MAE", f"{s['mae']:.2f}", "mg/dL · lower is better", "teal", ""),
                ("Model RMSE", f"{s['rmse']:.2f}", "mg/dL · penalizes large misses", "teal", ""),
            ]
            if "b_mae" in s:
                items += [("Baseline MAE", f"{s['b_mae']:.2f}", "mg/dL · persistence", "indigo", ""),
                          ("Baseline RMSE", f"{s['b_rmse']:.2f}", "mg/dL", "indigo", "")]
                if s.get("change") is not None:
                    up = s["change"] >= 0
                    items.append(("MAE vs baseline", f"{s['change']:+.1f}%",
                                  "▲ model error lower" if up else "▼ model error higher",
                                  "green" if up else "rose", "kpi-up" if up else "kpi-down"))
            else:
                items.append(("Rows evaluated", f"{s['n']:,}", "No baseline column found", "amber", ""))
            kpi_row(items)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            xcol = timestamp_x(work)
            x = work[xcol] if xcol else np.arange(1, len(work) + 1)
            v1, v2, v3 = st.tabs(["Timeline", "Actual vs predicted", "Error distribution"])
            with v1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=work[a_col], name="Actual", mode="lines", line=dict(color=C["ink"], width=2.4)))
                fig.add_trace(go.Scatter(x=x, y=work[p_col], name="Model", mode="lines", line=dict(color=C["teal"], width=2.2)))
                if b_col and work[b_col].notna().any():
                    fig.add_trace(go.Scatter(x=x, y=work[b_col], name="Baseline", mode="lines",
                                             line=dict(color="#9277b5", width=1.8, dash="dash")))
                plot_style(fig, "Glucose (mg/dL)", xcol or "Evaluation row", height=410)
                st.plotly_chart(fig, use_container_width=True)
            with v2:
                lo = float(min(work[a_col].min(), work[p_col].min()))
                hi = float(max(work[a_col].max(), work[p_col].max()))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                         line=dict(color="#94a3b8", dash="dot")))
                fig.add_trace(go.Scatter(x=work[a_col], y=work[p_col], mode="markers", name="Model",
                                         marker=dict(color=C["teal"], size=6, opacity=.6)))
                plot_style(fig, "Predicted (mg/dL)", "Actual (mg/dL)", height=410)
                fig.update_layout(hovermode="closest")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points close to the dotted diagonal indicate predictions close to the observed values.")
            with v3:
                err = work[p_col] - work[a_col]
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=err, nbinsx=30, marker_color=C["teal"], opacity=.85, name="Model error"))
                if b_col and work[b_col].notna().any():
                    fig.add_trace(go.Histogram(x=(work[b_col] - work[a_col]).dropna(), nbinsx=30,
                                               marker_color="#9277b5", opacity=.45, name="Baseline error"))
                fig.update_layout(barmode="overlay")
                fig.add_vline(x=0, line_dash="dot", line_color="#64748b")
                plot_style(fig, "Count", "Prediction error (mg/dL)", height=410)
                fig.update_layout(hovermode="closest")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Mean error (bias): {err.mean():+.2f} mg/dL · Error SD: {err.std():.2f} mg/dL")
            with st.expander("View evaluation records"):
                show_table(work, f"{selected_patient}_forecast")
    st.caption("Retrospective results on synthetic data, not live forecasts or evidence of clinical performance.")

# ------------------------------ EVENT EVALUATION ---------------------------
with tabs[2]:
    section("Event-classification evaluation", "Review binary event predictions and common classification metrics.")
    if patient_events.empty:
        empty_state("No event rows", "No event evaluation rows match this patient.", "⚡")
        if events_error:
            st.caption(events_error)
        elif not events.empty:
            st.caption("Available event IDs: " + ", ".join(clean_ids(events)))
    else:
        valid, a_col, p_col, prob_col = event_prepare(patient_events)
        if valid is None:
            st.error("Event evaluation requires recognizable actual-label and predicted-label columns.")
            st.caption("Detected columns: " + ", ".join(patient_events.columns))
            show_table(patient_events, f"{selected_patient}_events")
        elif valid.empty:
            st.warning("No valid binary event labels were found. Supported values include 0/1, true/false, yes/no, positive/negative, and event/no_event.")
            show_table(patient_events, f"{selected_patient}_events")
        else:
            s = event_stats(valid)
            event_kpis(s)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            left, right = st.columns([1, 1.25])
            with left:
                st.markdown("#### Confusion matrix")
                z = [[s["tn"], s["fp"]], [s["fn"], s["tp"]]]
                fig = go.Figure(go.Heatmap(
                    z=z, x=["Predicted: no event", "Predicted: event"], y=["Actual: no event", "Actual: event"],
                    text=[[f"TN<br><b>{s['tn']}</b>", f"FP<br><b>{s['fp']}</b>"], [f"FN<br><b>{s['fn']}</b>", f"TP<br><b>{s['tp']}</b>"]],
                    texttemplate="%{text}", colorscale=[[0, "#f1f7f9"], [1, C["teal"]]], showscale=False,
                    hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
                ))
                plot_style(fig, height=330)
                fig.update_layout(hovermode="closest")
                fig.update_yaxes(autorange="reversed", showgrid=False)
                fig.update_xaxes(showgrid=False, showspikes=False)
                st.plotly_chart(fig, use_container_width=True)
            with right:
                st.markdown("#### Event score timeline")
                if prob_col:
                    pr = valid.copy()
                    pr[prob_col] = numeric(pr[prob_col])
                    pr = pr.dropna(subset=[prob_col])
                    if not pr.empty:
                        xcol = timestamp_x(pr)
                        x = pr[xcol] if xcol else np.arange(1, len(pr) + 1)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=pr[prob_col], name="Event score", mode="lines",
                                                 line=dict(color=C["teal"], width=2.3), fill="tozeroy", fillcolor="rgba(14,154,167,.08)"))
                        pos = pr[pr["_a"] == 1]
                        if not pos.empty:
                            px_ = pos[xcol] if xcol else pos.index
                            fig.add_trace(go.Scatter(x=px_, y=pos[prob_col], mode="markers", name="Actual event",
                                                     marker=dict(color=C["rose"], size=7, line=dict(color="white", width=1))))
                        fig.add_hline(y=.5, line_dash="dash", line_color="#9277b5", annotation_text="0.5 reference")
                        plot_style(fig, "Model score", xcol or "Evaluation row", height=330)
                        fig.update_yaxes(range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        empty_state("No numeric scores", "The event-score column has no numeric values.", "📉")
                else:
                    empty_state("No score column", "No probability/score column is available for a timeline.", "📉")
            with st.expander("View event evaluation records"):
                show_table(valid.drop(columns=["_a", "_p"], errors="ignore"), f"{selected_patient}_events")
    notice("Event metrics are exploratory synthetic results. This classifier is not a medical alert system and must not guide treatment decisions.")

# ------------------------------- DATA EXPLORER -----------------------------
with tabs[3]:
    section("Data explorer", "Inspect source datasets, filter by the selected profile, and export records for reproducibility.")
    datasets = {label: (d, e) for label, d, e in SOURCES}
    c1, c2, c3 = st.columns([1.2, 1, 1.4])
    with c1:
        selected_dataset = st.selectbox("Dataset", list(datasets.keys()), key="explorer_dataset")
    source_df, source_error = datasets[selected_dataset]
    with c2:
        filter_patient = st.toggle("Selected patient only", value=True, key="explorer_patient_filter")
    with c3:
        search_text = st.text_input("Search values", placeholder="Optional keyword…", key="explorer_search")
    if source_df.empty:
        empty_state("Dataset unavailable", f"No data is available for {selected_dataset.lower()}.", "🗃️")
        if source_error:
            st.caption(source_error)
    else:
        has_id = find_col(source_df, ID_NAMES)
        visible = subset_patient(source_df, selected_patient) if filter_patient and selected_patient and has_id else source_df.copy()
        if search_text.strip() and not visible.empty:
            mask = visible.astype(str).apply(lambda col: col.str.contains(search_text.strip(), case=False, na=False, regex=False)).any(axis=1)
            visible = visible.loc[mask]
        if visible.empty:
            empty_state("No matching rows", "Try clearing the search or turning off the patient filter.", "🔍")
        else:
            st.caption(f"{selected_dataset} · {len(visible):,} rows · {len(visible.columns):,} columns")
            st.dataframe(pretty_df(visible), hide_index=True, use_container_width=True, height=430)
            st.download_button("⬇ Download filtered dataset", visible.to_csv(index=False).encode("utf-8"),
                               f"{selected_dataset.lower().replace(' ', '_')}.csv", "text/csv", key="explorer_download")
            with st.expander("Technical field reference"):
                ref = pd.DataFrame({
                    "Display name": [str(c).replace("_", " ").title() for c in visible.columns],
                    "Original column": list(visible.columns),
                    "Data type": [str(visible[c].dtype) for c in visible.columns],
                    "Non-null values": [int(visible[c].notna().sum()) for c in visible.columns],
                })
                st.dataframe(ref, hide_index=True, use_container_width=True)

# ------------------------------- MODEL SUMMARY -----------------------------
with tabs[4]:
    section("Model summary", "A cross-patient summary of the available retrospective evaluation outputs.")
    st.markdown("#### Forecasting")
    if not forecast.empty:
        f, a_col, p_col, b_col = forecast_prepare(forecast)
        if f is None:
            st.info("Forecast file loaded, but recognizable actual/prediction columns were not found.")
            st.caption("Detected columns: " + ", ".join(forecast.columns))
        elif f.empty:
            st.info("Forecast data loaded, but no complete numeric rows are available.")
        else:
            s = forecast_stats(f, a_col, p_col, b_col)
            items = [("Forecast MAE", f"{s['mae']:.2f}", "mg/dL", "teal", ""),
                     ("Forecast RMSE", f"{s['rmse']:.2f}", "mg/dL", "teal", ""),
                     ("Evaluation rows", f"{s['n']:,}", "Across all patients", "indigo", "")]
            if "b_mae" in s:
                items.append(("Baseline MAE", f"{s['b_mae']:.2f}", "mg/dL", "indigo", ""))
                if s.get("change") is not None:
                    up = s["change"] >= 0
                    items.append(("MAE vs baseline", f"{s['change']:+.1f}%", "▲ lower error" if up else "▼ higher error",
                                  "green" if up else "rose", "kpi-up" if up else "kpi-down"))
            kpi_row(items)
            id_col = find_col(f, ID_NAMES)
            if id_col and f[id_col].nunique() > 1:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                per = (f.assign(_ae=(f[p_col] - f[a_col]).abs()).groupby(f[id_col].astype(str))["_ae"].mean().sort_values())
                fig = go.Figure(go.Bar(
                    x=per.values, y=per.index, orientation="h",
                    marker_color=[C["amber"] if i == selected_patient else C["teal"] for i in per.index],
                    hovertemplate="%{y}: %{x:.2f} mg/dL<extra></extra>",
                ))
                plot_style(fig, "", "MAE (mg/dL)", height=max(260, 28 * len(per) + 80))
                fig.update_layout(hovermode="closest", title=dict(text="MAE by patient (selected patient highlighted)", font=dict(size=13)))
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
    else:
        empty_state("No forecast data", forecast_error or "Forecast evaluation data is empty.", "🎯")

    st.markdown("#### Event classification")
    if not events.empty:
        ev, _, _, _ = event_prepare(events)
        if ev is None:
            st.info("Event file loaded, but recognizable actual/predicted columns were not found.")
        elif ev.empty:
            st.info("No valid binary event rows are available for summary metrics.")
        else:
            event_kpis(event_stats(ev))
    else:
        empty_state("No event data", events_error or "Event evaluation data is empty.", "⚡")

    with st.expander("How to interpret these metrics", expanded=False):
        st.write(
            "Lower MAE and RMSE indicate smaller retrospective forecast error. "
            "Precision and recall capture different aspects of event classification: precision asks how many predicted events were real, "
            "recall asks how many real events were found. These metrics depend on the synthetic dataset and do not establish real-world or clinical performance."
        )

# ---------------------------------- ABOUT ----------------------------------
with tabs[5]:
    section("About GlucoTwin", "Project scope, reproducibility, data availability, and responsible-use notes.")
    a1, a2 = st.columns([1.3, 1])
    with a1:
        st.markdown("#### Project purpose")
        st.write("GlucoTwin is a research prototype for exploring synthetic patient profiles, simulated glucose time series, retrospective forecasting evaluation, and binary event-classification experiments.")
        st.markdown("#### Limitations and responsible use")
        st.write("All records in this demo are synthetic. Retrospective metrics do not establish clinical validity. The app is not a medical device and must not be used for diagnosis, treatment decisions, or real-world medical alerts.")
        st.markdown("#### Project structure")
        st.code("GlucoTwin/\n├── app.py\n├── requirements.txt\n└── data/\n    ├── synthetic_ehr.csv\n    ├── synthetic_timeseries.csv\n    ├── forecast_evaluation.csv\n    └── event_evaluation.csv", language="text")
    with a2:
        st.markdown("#### Dataset health")
        for label, d, e in SOURCES:
            status_row(label, d, e, dark=False)

st.markdown(
    '<div class="gt-footer"><b>GlucoTwin · Research Intelligence</b><br>Digital health research prototype · Synthetic data only<br>'
    'Not for diagnosis, treatment decisions, or real-world medical alerts.</div>',
    unsafe_allow_html=True,
)