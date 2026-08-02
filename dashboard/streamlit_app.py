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

The "AI Assistant" tab is a chat console over the same rule-based (no LLM)
engine as the Flask dashboard - dashboard/chatbot.py - plus a structured
prediction card for whichever transformer the conversation last mentioned.

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
import knowledge_base  # noqa: E402
import outage_forecasting as of  # noqa: E402
import report  # noqa: E402
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
    /* Eskom-blue diagonal wash behind the whole app. Every chart, tile, and
    card below sets its own opaque surface color (#fcfcfb), so this never
    shows through data - it only fills the space around it. */
    [data-testid="stAppViewContainer"], .stApp {
        background: linear-gradient(135deg, #9dc4f2 0%, #dceafb 20%, #ffffff 46%, #ffffff 54%, #dceafb 80%, #9dc4f2 100%);
        background-attachment: fixed;
    }
    .sg-header {
        display:flex; align-items:center; gap:14px; margin-bottom: 6px;
        position:relative; overflow:hidden; padding:10px 14px; border-radius:14px;
    }
    .sg-header::before {
        content:""; position:absolute; top:-120%; right:-10%; width:42%; height:340%;
        background: linear-gradient(135deg, #2a78d6 0%, #5aa0e6 55%, rgba(90,160,230,0) 100%);
        clip-path: polygon(35% 0%, 100% 0%, 100% 100%, 0% 100%);
        opacity:0.26; pointer-events:none; z-index:0;
    }
    .sg-header > * { position:relative; z-index:1; }
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
    .sg-card {
        border:1px solid rgba(11,11,11,0.10); border-top:3px solid var(--sg-accent, #2a78d6);
        border-radius:10px; padding:16px 18px; background:#fcfcfb;
    }
    .sg-card h4 { margin:0 0 12px 0; font-size:15px; font-weight:800; }
    .sg-card .sg-tag {
        display:inline-block; font-size:11px; font-weight:700; padding:2px 9px;
        border-radius:999px; color:#fff; margin-left:8px; vertical-align:middle;
    }
    .sg-field-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px 16px; margin-bottom:14px; }
    .sg-field .sg-field-value { font-size:19px; font-weight:800; font-variant-numeric: tabular-nums; }
    .sg-field .sg-field-label { font-size:11.5px; color:#75736c; margin-top:1px; }
    .sg-check-list { margin:6px 0 4px 0; padding:0; list-style:none; }
    .sg-check-list li { font-size:13.5px; padding:3px 0; color:#3a3934; }
    .sg-check-list li::before { content:"✔"; color:#0ca30c; margin-right:8px; font-weight:800; }
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


CHART_FONT = dict(family="-apple-system, Segoe UI, sans-serif", color="#3a3934", size=12.5)


def tier_bar_chart(scores_df, tier_col):
    counts = scores_df[tier_col].value_counts().reindex(TIER_ORDER, fill_value=0)
    total = counts.values.sum()
    pct = (counts.values / total * 100) if total else counts.values * 0
    fig = go.Figure(
        go.Bar(
            x=[t.title() for t in TIER_ORDER],
            y=counts.values,
            marker=dict(color=STATUS_COLORS, cornerradius=6),
            text=counts.values,
            textposition="outside",
            cliponaxis=False,
            customdata=pct,
            hovertemplate="<b>%{x}</b><br>%{y:,} assets · %{customdata:.1f}%<extra></extra>",
            width=0.55,
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=30, r=10, t=10, b=30),
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        showlegend=False,
        font=CHART_FONT,
        hoverlabel=dict(bgcolor="#232220", font_color="#fff", bordercolor="#232220"),
        yaxis=dict(showgrid=True, gridcolor="#e9e8e2", gridwidth=1, range=[0, counts.values.max() * 1.25], zeroline=False),
        xaxis=dict(showgrid=False),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def importance_bar_chart(importance_df, accent):
    ordered = importance_df.sort_values("importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered["importance"],
            y=ordered["feature"],
            orientation="h",
            marker=dict(color=accent, cornerradius=5),
            text=ordered["importance"].round(3),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>importance %{x:.3f}<extra></extra>",
            width=0.55,
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=30, t=10, b=30),
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        showlegend=False,
        font=CHART_FONT,
        hoverlabel=dict(bgcolor="#232220", font_color="#fff", bordercolor="#232220"),
        xaxis=dict(showgrid=True, gridcolor="#e9e8e2", gridwidth=1, zeroline=False),
        bargap=0.35,
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
            marker=dict(color=colors, cornerradius=5),
            text=[f"{v:+.2f}" for v in ordered.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>contribution %{x:+.2f}<extra></extra>",
            width=0.55,
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=30, t=10, b=30),
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        showlegend=False,
        font=CHART_FONT,
        hoverlabel=dict(bgcolor="#232220", font_color="#fff", bordercolor="#232220"),
        xaxis=dict(showgrid=True, gridcolor="#e9e8e2", gridwidth=1, zeroline=True, zerolinecolor="#c3c2b7"),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("🔴 pushes the score up · 🔵 pushes the score down")


def risk_landscape_3d(df, x_col, y_col, z_col, tier_col, id_col, labels, key):
    """Rotatable 3D scatter of the real scored fleet - x/y are two raw model
    features, z is the model's own risk/anomaly score, colored by the same
    fixed severity ramp used everywhere else (never a second, unrelated
    color scheme) so this reads as one more view of the same prediction,
    not a decorative extra."""
    sample = df if len(df) <= 1200 else df.sample(n=1200, random_state=42)
    fig = go.Figure()
    for tier, color in zip(TIER_ORDER, STATUS_COLORS):
        sub = sample[sample[tier_col] == tier]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=sub[x_col], y=sub[y_col], z=sub[z_col],
                mode="markers",
                name=tier.title(),
                marker=dict(size=3.5, color=color, opacity=0.75, line=dict(width=0)),
                customdata=sub[[id_col]],
                hovertemplate=(
                    f"<b>%{{customdata[0]}}</b><br>{labels[0]}: %{{x:.2f}}<br>"
                    f"{labels[1]}: %{{y:.2f}}<br>{labels[2]}: %{{z:.1f}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#fcfcfb",
        font=CHART_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1, x=0),
        scene=dict(
            xaxis=dict(title=labels[0], backgroundcolor="#fcfcfb", gridcolor="#e9e8e2"),
            yaxis=dict(title=labels[1], backgroundcolor="#fcfcfb", gridcolor="#e9e8e2"),
            zaxis=dict(title=labels[2], backgroundcolor="#fcfcfb", gridcolor="#e9e8e2"),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
    if len(df) > 1200:
        st.caption(f"Showing a random sample of 1,200 of {len(df):,} for a responsive plot - drag to rotate.")
    else:
        st.caption("Drag to rotate, scroll to zoom.")


def tier_badge_table(scores_df, id_col, score_col, tier_col, extra_cols, n=15):
    top = scores_df.sort_values(score_col, ascending=False).head(n).copy()
    cols = [id_col, score_col, tier_col] + extra_cols
    st.dataframe(top[cols], use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# AI Assistant tab - structured prediction card, history, and compare views.
# All figures come from the same merged raw+scores frame the rest of the app
# uses (get_transformer_data), never invented for display.
# --------------------------------------------------------------------------
STATUS_COLOR_BY_TIER = dict(zip(TIER_ORDER, STATUS_COLORS))


def get_transformer_full():
    raw, scores = get_transformer_data()
    return scores.merge(raw, on="transformer_id")


@st.cache_data
def get_transformer_history():
    return pd.read_csv(ROOT / "data" / "transformer_history.csv")


def render_prediction_card(transformer_id):
    full = get_transformer_full()
    match = full[full["transformer_id"] == transformer_id]
    if match.empty:
        st.info(f"No score data for {transformer_id} yet - run models/failure_prediction.py first.")
        return None
    row = match.iloc[0]
    tier = row["risk_tier"]
    tag_color = STATUS_COLOR_BY_TIER.get(tier, "#52514e")
    actions = knowledge_base.MAINTENANCE_ACTIONS.get(tier, [])
    checklist = "".join(f"<li>{a}</li>" for a in actions) or "<li>No recommendation available.</li>"

    st.markdown(
        f"""
        <div class="sg-card" style="--sg-accent:{ACCENT['transformer']}">
          <h4>📋 Prediction Result — {row['transformer_id']}
            <span class="sg-tag" style="background:{tag_color}">{tier.upper()}</span>
          </h4>
          <div class="sg-field-grid">
            <div class="sg-field"><div class="sg-field-value">{row['health_score']:.0f}%</div><div class="sg-field-label">Health score</div></div>
            <div class="sg-field"><div class="sg-field-value">{tier.title()}</div><div class="sg-field-label">Risk level</div></div>
            <div class="sg-field"><div class="sg-field-value">{row['risk_score']:.0f}%</div><div class="sg-field-label">Failure probability</div></div>
            <div class="sg-field"><div class="sg-field-value">{row['confidence_pct']:.0f}%</div><div class="sg-field-label">Confidence</div></div>
            <div class="sg-field" style="grid-column:1 / -1;"><div class="sg-field-value" style="font-size:15px;">{row['predicted_failure_mode']}</div><div class="sg-field-label">Likely cause</div></div>
          </div>
          <div style="font-size:12.5px; font-weight:700; color:#52514e; margin-bottom:2px;">Recommendations</div>
          <ul class="sg-check-list">{checklist}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return row


def render_history(row):
    hist_all = get_transformer_history()
    hist = hist_all[hist_all["transformer_id"] == row["transformer_id"]].sort_values("month_offset")
    if hist.empty:
        st.caption("No history data for this transformer.")
        return
    st.markdown(f"**Maintenance history — {row['transformer_id']}**")
    st.caption(
        f"Last serviced {row['last_serviced_date']} · last oil replacement {row['last_oil_replacement_date']} · "
        f"{int(row['previous_failures'])} previous failure(s)"
    )
    metrics = [
        ("oil_quality_index", "Oil quality index", ACCENT["transformer"]),
        ("temperature_rise_c", "Temperature rise (°C)", CONTRIB_UP),
        ("load_factor", "Load factor", ACCENT["feeder"]),
    ]
    cols = st.columns(3)
    for col, (metric, label, color) in zip(cols, metrics):
        fig = go.Figure(
            go.Scatter(
                x=hist["month_offset"], y=hist[metric], mode="lines+markers",
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color, line=dict(width=2, color="#fcfcfb")),
                hovertemplate=f"month %{{x}}<br>{label}: %{{y:.2f}}<extra></extra>",
            )
        )
        fig.update_layout(
            height=180, margin=dict(l=10, r=10, t=24, b=24),
            paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
            font=CHART_FONT, showlegend=False, title=dict(text=label, font=dict(size=12)),
            xaxis=dict(showgrid=False, title="Month"),
            yaxis=dict(showgrid=True, gridcolor="#e9e8e2", zeroline=False),
        )
        col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_compare(row):
    full = get_transformer_full()
    pool = full[full["transformer_id"] != row["transformer_id"]].sort_values("risk_score", ascending=False).head(30)
    other_id = st.selectbox(
        "Compare against (top 30 by risk)", pool["transformer_id"].tolist(),
        key=f"compare_pick_{row['transformer_id']}",
    )
    other = full[full["transformer_id"] == other_id].iloc[0]

    st.markdown(f"**{row['transformer_id']} vs {other_id}**")
    metrics = [
        ("Risk score", "risk_score", "{:.1f}%"),
        ("Health score", "health_score", "{:.1f}%"),
        ("Confidence", "confidence_pct", "{:.1f}%"),
        ("Temperature rise", "temperature_rise_c", "{:.1f}°C"),
        ("Load factor", "load_factor", "{:.2f}"),
        ("Predicted failure", "predicted_failure_mode", "{}"),
    ]
    table_rows = [[label, fmt.format(row[col]), fmt.format(other[col])] for label, col, fmt in metrics]
    st.dataframe(
        pd.DataFrame(table_rows, columns=["Metric", row["transformer_id"], other_id]),
        use_container_width=True, hide_index=True,
    )

    bar_metrics, bar_labels = ["risk_score", "health_score", "confidence_pct"], ["Risk score", "Health score", "Confidence"]
    fig = go.Figure([
        go.Bar(name=row["transformer_id"], x=bar_labels, y=[row[m] for m in bar_metrics],
               marker=dict(color=ACCENT["transformer"], cornerradius=6)),
        go.Bar(name=other_id, x=bar_labels, y=[other[m] for m in bar_metrics],
               marker=dict(color=ACCENT["meter"], cornerradius=6)),
    ])
    fig.update_layout(
        barmode="group", height=260, margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=CHART_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(showgrid=True, gridcolor="#e9e8e2", range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


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
    st.caption("Chat with the AI Assistant tab for Q&A, predictions, and per-transformer reports.")

tab_assistant, tab_failure, tab_theft, tab_outage = st.tabs(
    ["🤖 AI Assistant", "Transformer Failure Prediction", "Electricity Theft Detection", "Feeder Outage Forecasting"]
)

# --------------------------------------------------------------------------
# AI Assistant - an engineer's assistant for asset health, not a support
# bot: chat on the left drives predict/health/history/compare intents over
# the real scored fleet (dashboard/chatbot.py), the right panel renders the
# focused transformer's own prediction as a structured card, matching what
# the model actually output - never restated or re-derived for display.
# --------------------------------------------------------------------------
with tab_assistant:
    st.markdown(
        """
        <div class="sg-header">
          <div class="sg-brand-mark">AI</div>
          <div>
            <h1>PredictAI Transformer Assistant</h1>
            <p>Ask about asset health, interpret predictions, diagnose faults, and get maintenance recommendations - rule-based over the scored fleet, no LLM.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("chat_context", {})

    chat_col, card_col = st.columns([1.15, 1])

    with chat_col:
        header_col, clear_col = st.columns([3, 1])
        header_col.markdown("#### 💬 Chat")
        if clear_col.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_context = {}
            for key in ("assistant_show_history", "assistant_show_compare", "assistant_report_path", "assistant_focus_id"):
                st.session_state.pop(key, None)

        pending_question = st.chat_input("e.g. Predict transformer T0208")

        chips = [
            "Predict T0208",
            "Health of T0208",
            "Why is T0208 high risk?",
            "Top 5 riskiest feeders",
        ]
        chip_cols = st.columns(2)
        for i, chip in enumerate(chips):
            if chip_cols[i % 2].button(chip, key=f"chip_{i}", use_container_width=True):
                pending_question = chip

        if pending_question:
            st.session_state.chat_history.append({"role": "user", "content": pending_question})
            reply = chatbot.answer(pending_question, ROOT / "data", context=st.session_state.chat_context)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        if not st.session_state.chat_history:
            st.caption("Ask me anything about the scored fleet, or tap a suggestion above.")

        with st.container(height=440):
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

    with card_col:
        last_id = st.session_state.chat_context.get("last_id")
        focus_id = None
        if last_id and last_id[0] == "transformer":
            full = get_transformer_full()
            id_nums = full["transformer_id"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
            match = full[id_nums == last_id[2]]
            if not match.empty:
                focus_id = match.iloc[0]["transformer_id"]

        if focus_id != st.session_state.get("assistant_focus_id"):
            for key in ("assistant_show_history", "assistant_show_compare", "assistant_report_path"):
                st.session_state.pop(key, None)
            st.session_state["assistant_focus_id"] = focus_id

        if focus_id is None:
            st.markdown("#### 📋 Prediction Result")
            st.caption("Ask about a specific transformer (e.g. “predict T0208”) to see its full prediction card here.")
        else:
            row = render_prediction_card(focus_id)
            if row is not None:
                b1, b2, b3 = st.columns(3)
                if b1.button("📄 Generate PDF Report", use_container_width=True, key=f"pdf_{focus_id}"):
                    path = report.generate_pdf_report(focus_id, ROOT / "data")
                    st.session_state["assistant_report_path"] = str(path)
                if b2.button("📈 View History", use_container_width=True, key=f"hist_{focus_id}"):
                    st.session_state["assistant_show_history"] = not st.session_state.get("assistant_show_history", False)
                if b3.button("⚖️ Compare", use_container_width=True, key=f"cmp_{focus_id}"):
                    st.session_state["assistant_show_compare"] = not st.session_state.get("assistant_show_compare", False)

                if st.session_state.get("assistant_report_path"):
                    report_path = Path(st.session_state["assistant_report_path"])
                    if report_path.exists():
                        st.download_button(
                            f"⬇ Download {report_path.name}", report_path.read_bytes(), report_path.name,
                            mime="application/pdf", key=f"dl_{focus_id}", use_container_width=True,
                        )

                if st.session_state.get("assistant_show_history"):
                    render_history(row)

                if st.session_state.get("assistant_show_compare"):
                    render_compare(row)

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

    st.subheader("3D risk landscape")
    st.caption("Age × temperature rise × predicted risk score, one point per transformer")
    landscape = raw.merge(scores[["transformer_id", "risk_score", "risk_tier"]], on="transformer_id")
    risk_landscape_3d(
        landscape, "age_years", "temperature_rise_c", "risk_score", "risk_tier", "transformer_id",
        ["Age (years)", "Temp rise (°C)", "Risk score"], key="landscape_transformer",
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

    st.subheader("3D risk landscape")
    st.caption("Recent usage drop × night usage ratio × anomaly score, one point per meter")
    landscape = raw.merge(scores[["meter_id", "anomaly_score", "priority_tier"]], on="meter_id")
    risk_landscape_3d(
        landscape, "pct_drop_recent", "night_usage_ratio", "anomaly_score", "priority_tier", "meter_id",
        ["Recent usage drop (%)", "Night usage ratio", "Anomaly score"], key="landscape_meter",
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

    st.subheader("3D risk landscape")
    st.caption("Peak load × vegetation encroachment × outage risk score, one point per feeder")
    landscape = raw.merge(scores[["feeder_id", "outage_risk_score", "risk_tier"]], on="feeder_id")
    risk_landscape_3d(
        landscape, "peak_load_pct", "vegetation_encroachment_score", "outage_risk_score", "risk_tier", "feeder_id",
        ["Peak load (%)", "Vegetation score", "Outage risk score"], key="landscape_feeder",
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
