"""
SmartGrid PredictAI - Prediction Console (Streamlit)
======================================================
Auto-loads the same data/*.csv files the Flask dashboard (dashboard/app.py)
reads - no CSV upload step - scores all three models (transformer failure,
meter theft, feeder outage) and shows per-entity results plus a SHAP-based
"why" for any one you pick. Falls back to training on the spot if a
data/*_scores.csv file doesn't exist yet, so a clean checkout still works;
prefers the already-trained models/*.joblib and pre-scored CSVs produced by
running models/failure_prediction.py, models/theft_detection.py, and
models/outage_forecasting.py.

A sidebar chat panel answers questions about the scored data using the same
rule-based (no LLM) engine as the Flask dashboard - dashboard/chatbot.py.

Run from the repo root:
    streamlit run dashboard/streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))

import chatbot  # noqa: E402  (dashboard/chatbot.py, same directory as this file)
import explain  # noqa: E402
import failure_prediction as fp  # noqa: E402
import outage_forecasting as of  # noqa: E402
import theft_detection as td  # noqa: E402

st.set_page_config(page_title="SmartGrid PredictAI", layout="wide", page_icon="⚡")

# Same brand palette as dashboard/static/style.css, so the two apps read as
# one product: blue/orange/aqua for transformer/meter/feeder identity (the
# categorical palette's first three slots), plus the fixed four-tier status
# ramp (never reused for anything else) and a blue/red diverging pair
# reserved for SHAP contribution direction (increases vs decreases risk).
ACCENT = {"transformer": "#2a78d6", "meter": "#eb6834", "feeder": "#1baf7a"}
STATUS_COLORS = ["#0ca30c", "#fab219", "#ec835a", "#d03b3b"]  # low, moderate, elevated, critical
TIER_ORDER = ["low", "moderate", "elevated", "critical"]
CONTRIB_UP, CONTRIB_DOWN = "#e34948", "#2a78d6"  # pushes score up / down

st.markdown(
    """
    <style>
    .sg-header { display:flex; align-items:center; gap:14px; margin-bottom: 6px; }
    .sg-brand-mark {
        width:44px; height:44px; border-radius:12px; flex-shrink:0;
        display:flex; align-items:center; justify-content:center;
        font-weight:800; font-size:15px; color:#fff;
        background: linear-gradient(135deg, #2a78d6, #1baf7a);
        box-shadow: 0 6px 16px rgba(42,120,214,0.28);
    }
    .sg-header h1 { margin:0; font-size:23px; font-weight:800; letter-spacing:-0.01em; }
    .sg-header p { margin:0; color:#52514e; font-size:14px; }
    .sg-tile {
        border:1px solid rgba(11,11,11,0.10); border-top:3px solid var(--sg-accent, #2a78d6);
        border-radius:8px; padding:12px 14px; background:#fcfcfb;
    }
    .sg-tile .sg-value { font-size:25px; font-weight:800; letter-spacing:-0.01em; font-variant-numeric: tabular-nums; }
    .sg-tile .sg-label { font-size:12px; color:#52514e; margin-top:2px; }
    .sg-caption { color:#898781; font-size:12.5px; margin-top:-6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached model loaders - prefer the joblib artifacts the scripts save, train
# on the spot only if they don't exist yet.
# --------------------------------------------------------------------------
@st.cache_resource
def load_failure_model():
    model_path = ROOT / "models" / "failure_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    df = fp.load_data(str(ROOT / "data" / "transformer_data.csv"))
    model, _ = fp.train_model(df)
    return model


@st.cache_resource
def load_theft_model():
    model_path = ROOT / "models" / "theft_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    df = td.load_data(str(ROOT / "data" / "meter_data.csv"))
    model, scaler, feature_cols, _, _ = td.train_model(df)
    return {"model": model, "scaler": scaler, "features": feature_cols}


@st.cache_resource
def load_outage_model():
    model_path = ROOT / "models" / "outage_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    df = of.load_data(
        str(ROOT / "data" / "feeder_data.csv"), str(ROOT / "data" / "transformer_risk_scores.csv")
    )
    return of.train_model(df)


# --------------------------------------------------------------------------
# Cached auto-loaders - read the pre-scored CSV if it exists (produced by
# running the scripts), otherwise score now and write it so dependent data
# (outage forecasting reads transformer_risk_scores.csv) is available too.
# --------------------------------------------------------------------------
@st.cache_data
def get_transformer_data():
    raw = fp.load_data(str(ROOT / "data" / "transformer_data.csv"))
    scores_path = ROOT / "data" / "transformer_risk_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
    else:
        scores = fp.score_all_transformers(load_failure_model(), raw, explain_predictions=True)
        scores.to_csv(scores_path, index=False)
    return raw, scores


@st.cache_data
def get_meter_data():
    raw = td.load_data(str(ROOT / "data" / "meter_data.csv"))
    scores_path = ROOT / "data" / "meter_theft_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
    else:
        bundle = load_theft_model()
        model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]
        X_scaled = scaler.transform(raw[feature_cols])
        raw_scores = -model.score_samples(X_scaled)
        anomaly_score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9) * 100
        flagged = model.predict(X_scaled) == -1
        scores = td.score_all_meters(
            raw, scaler, feature_cols, anomaly_score, flagged, model=model, explain_predictions=True
        )
        scores.to_csv(scores_path, index=False)
    return raw, scores


@st.cache_data
def get_feeder_data():
    get_transformer_data()  # ensures transformer_risk_scores.csv exists on disk first
    raw = of.load_data(
        str(ROOT / "data" / "feeder_data.csv"), str(ROOT / "data" / "transformer_risk_scores.csv")
    )
    scores_path = ROOT / "data" / "feeder_outage_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
    else:
        scores = of.score_all_feeders(load_outage_model(), raw, explain_predictions=True)
        scores.to_csv(scores_path, index=False)
    return raw, scores


@st.cache_data
def get_meter_importance():
    # Isolation Forest has no feature_importances_, so global importance
    # comes from SHAP contributions on a sample (fast, statistically fine
    # for an aggregate ranking - the per-row explainer below uses the exact
    # row, not the sample).
    raw, _ = get_meter_data()
    bundle = load_theft_model()
    model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]
    sample = raw.sample(n=min(2000, len(raw)), random_state=42)
    X_scaled = pd.DataFrame(scaler.transform(sample[feature_cols]), columns=feature_cols, index=sample.index)
    contributions, _ = explain.explain_batch(model, X_scaled, feature_cols)
    return explain.importance_from_contributions(contributions)


def three_way_status(score, low_label, mid_label, high_label):
    if score < 40:
        return low_label
    if score < 75:
        return mid_label
    return high_label


# --------------------------------------------------------------------------
# Shared chart / layout helpers
# --------------------------------------------------------------------------
def render_stat_tiles(items, accent):
    cols = st.columns(len(items))
    for col, (value, label) in zip(cols, items):
        col.markdown(
            f'<div class="sg-tile" style="--sg-accent:{accent}">'
            f'<div class="sg-value">{value}</div><div class="sg-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


def tier_bar_chart(scores_df, tier_col):
    counts = scores_df[tier_col].value_counts().reindex(TIER_ORDER, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=[t.title() for t in TIER_ORDER],
            y=counts.values,
            marker_color=STATUS_COLORS,
            text=counts.values,
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=30, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor="#e1e0d9", range=[0, counts.values.max() * 1.2]),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def importance_bar_chart(importance_df, accent):
    ordered = importance_df.sort_values("importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered["importance"],
            y=ordered["feature"],
            orientation="h",
            marker_color=accent,
            text=ordered["importance"].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=30, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#e1e0d9"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def contribution_bar_chart(contributions_row):
    ordered = contributions_row.reindex(contributions_row.abs().sort_values().index)
    colors = [CONTRIB_UP if v >= 0 else CONTRIB_DOWN for v in ordered.values]
    fig = go.Figure(
        go.Bar(
            x=ordered.values,
            y=ordered.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in ordered.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=30, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#e1e0d9", zeroline=True, zerolinecolor="#c3c2b7"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"🔴 pushes the score up · 🔵 pushes the score down")


def tier_badge_table(scores_df, id_col, score_col, tier_col, extra_cols, n=15):
    top = scores_df.sort_values(score_col, ascending=False).head(n).copy()
    cols = [id_col, score_col, tier_col] + extra_cols
    st.dataframe(top[cols], use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="sg-header">
      <div class="sg-brand-mark">SG</div>
      <div>
        <h1>SmartGrid PredictAI</h1>
        <p>Auto-scored transformer failure, meter theft, and feeder outage risk - refreshes from data/*.csv, no upload needed.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Data")
    st.caption("Reads data/*.csv directly. Re-run the model scripts (or click below) to refresh.")
    if st.button("Refresh predictions", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    header_col, clear_col = st.columns([3, 1])
    header_col.markdown("### Ask about the data")
    st.caption("Rule-based Q&A over the score files - no LLM, no external calls. Try “why is T0208 high risk?” or “top 5 riskiest feeders”.")

    st.session_state.setdefault("chat_history", [])
    if clear_col.button("Clear", use_container_width=True):
        st.session_state.chat_history = []

    pending_question = st.chat_input("e.g. why is T0208 high risk?")

    chips = [
        "How many transformers are critical?",
        "Why are transformers critical?",
        "Which meters are flagged?",
        "Top 5 riskiest feeders",
    ]
    chip_cols = st.columns(2)
    for i, chip in enumerate(chips):
        if chip_cols[i % 2].button(chip, key=f"chip_{i}", use_container_width=True):
            pending_question = chip

    if pending_question:
        st.session_state.chat_history.append({"role": "user", "content": pending_question})
        reply = chatbot.answer(pending_question, ROOT / "data")
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if not st.session_state.chat_history:
        st.caption("Ask me anything about the scored fleet, or tap a suggestion above.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

tab_failure, tab_theft, tab_outage = st.tabs(
    ["Transformer Failure Prediction", "Electricity Theft Detection", "Feeder Outage Forecasting"]
)

# --------------------------------------------------------------------------
# Transformer failure prediction
# --------------------------------------------------------------------------
with tab_failure:
    raw, scores = get_transformer_data()
    scores = scores.copy()
    scores["status"] = scores["risk_score"].apply(
        lambda s: three_way_status(s, "Healthy", "High Risk", "Failure Likely")
    )
    counts = scores["status"].value_counts()

    render_stat_tiles(
        [
            (f"{len(scores):,}", "Total transformers"),
            (f"{int(counts.get('Failure Likely', 0)):,}", "Failure likely"),
            (f"{int(counts.get('High Risk', 0)):,}", "High risk"),
            (f"{scores['risk_score'].mean():.1f}", "Average risk score"),
        ],
        ACCENT["transformer"],
    )

    st.write("")
    left, right = st.columns([1, 1.4])
    with left:
        st.caption("Fleet breakdown by severity tier")
        tier_bar_chart(scores, "risk_tier")
    with right:
        st.caption("Top 15 highest-risk transformers")
        tier_badge_table(scores, "transformer_id", "risk_score", "risk_tier", ["status", "top_reasons"])

    st.download_button(
        "Download all predictions as CSV", scores.to_csv(index=False), "transformer_predictions.csv"
    )

    st.subheader("Why does the model predict this?")
    left, right = st.columns(2)
    with left:
        st.caption("Global feature importance (whole fleet)")
        model = load_failure_model()
        importance_bar_chart(explain.global_importance(model, fp.FEATURES), ACCENT["transformer"])
    with right:
        st.caption("Explain one transformer's prediction")
        pick_pool = scores.sort_values("risk_score", ascending=False).head(30)["transformer_id"].tolist()
        chosen_id = st.selectbox("Transformer (top 30 by risk)", pick_pool, key="failure_pick")
        row_df = raw[raw["transformer_id"] == chosen_id]
        contributions, used_shap = explain.explain_batch(model, row_df[fp.FEATURES], fp.FEATURES)
        contribution_bar_chart(contributions.iloc[0])
        if not used_shap:
            st.caption("Approximate contribution (shap not installed)")

# --------------------------------------------------------------------------
# Theft detection
# --------------------------------------------------------------------------
with tab_theft:
    raw, scores = get_meter_data()
    suspected = scores[scores["investigation_flag"] == 1]

    render_stat_tiles(
        [
            (f"{len(scores):,}", "Total meters"),
            (f"{len(suspected):,}", "Suspected theft cases"),
            (f"{int((scores['priority_tier'] == 'critical').sum()):,}", "Critical tier"),
            (f"{scores['anomaly_score'].mean():.1f}", "Average anomaly score"),
        ],
        ACCENT["meter"],
    )

    st.write("")
    left, right = st.columns([1, 1.4])
    with left:
        st.caption("Fleet breakdown by severity tier")
        tier_bar_chart(scores, "priority_tier")
    with right:
        st.caption("Suspected theft cases (highest anomaly score first)")
        tier_badge_table(
            suspected if len(suspected) else scores,
            "meter_id", "anomaly_score", "priority_tier", ["top_reasons"],
        )

    st.download_button(
        "Download all predictions as CSV", scores.to_csv(index=False), "meter_theft_predictions.csv"
    )

    st.subheader("Why does the model flag this?")
    left, right = st.columns(2)
    with left:
        st.caption("Global feature importance (sampled fleet)")
        importance_bar_chart(get_meter_importance(), ACCENT["meter"])
    with right:
        st.caption("Explain one meter's anomaly score")
        pick_pool = scores.sort_values("anomaly_score", ascending=False).head(30)["meter_id"].tolist()
        chosen_id = st.selectbox("Meter (top 30 by anomaly score)", pick_pool, key="theft_pick")
        bundle = load_theft_model()
        model, scaler, feature_cols = bundle["model"], bundle["scaler"], bundle["features"]
        row_df = raw[raw["meter_id"] == chosen_id]
        row_scaled = pd.DataFrame(
            scaler.transform(row_df[feature_cols]), columns=feature_cols, index=row_df.index
        )
        contributions, used_shap = explain.explain_batch(model, row_scaled, feature_cols)
        contribution_bar_chart(contributions.iloc[0])
        if not used_shap:
            st.caption("Approximate contribution (shap not installed)")

# --------------------------------------------------------------------------
# Outage forecasting
# --------------------------------------------------------------------------
with tab_outage:
    raw, scores = get_feeder_data()
    scores = scores.copy()
    scores["status"] = scores["outage_risk_score"].apply(
        lambda s: three_way_status(s, "Stable", "At Risk", "Outage Likely")
    )
    counts = scores["status"].value_counts()

    render_stat_tiles(
        [
            (f"{len(scores):,}", "Total feeders"),
            (f"{int(counts.get('Outage Likely', 0)):,}", "Outage likely"),
            (f"{int(counts.get('At Risk', 0)):,}", "At risk"),
            (f"{scores['outage_risk_score'].mean():.1f}", "Average outage risk score"),
        ],
        ACCENT["feeder"],
    )

    st.write("")
    left, right = st.columns([1, 1.4])
    with left:
        st.caption("Fleet breakdown by severity tier")
        tier_bar_chart(scores, "risk_tier")
    with right:
        st.caption("Top 15 highest-risk feeders")
        tier_badge_table(scores, "feeder_id", "outage_risk_score", "risk_tier", ["status", "top_reasons"])

    st.download_button(
        "Download all predictions as CSV", scores.to_csv(index=False), "feeder_outage_predictions.csv"
    )

    st.subheader("Why does the model predict this?")
    left, right = st.columns(2)
    with left:
        st.caption("Global feature importance (whole fleet)")
        model = load_outage_model()
        importance_bar_chart(explain.global_importance(model, of.FEATURES), ACCENT["feeder"])
    with right:
        st.caption("Explain one feeder's prediction")
        pick_pool = scores.sort_values("outage_risk_score", ascending=False).head(30)["feeder_id"].tolist()
        chosen_id = st.selectbox("Feeder (top 30 by risk)", pick_pool, key="outage_pick")
        row_df = raw[raw["feeder_id"] == chosen_id]
        contributions, used_shap = explain.explain_batch(model, row_df[of.FEATURES], of.FEATURES)
        contribution_bar_chart(contributions.iloc[0])
        if not used_shap:
            st.caption("Approximate contribution (shap not installed)")
