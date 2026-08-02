"""
SmartGrid PredictAI - Planner Dashboard
=========================================
Serves the script-pipeline outputs (data/transformer_risk_scores.csv,
data/meter_theft_scores.csv, data/feeder_outage_scores.csv) as a
planner-facing view: how many transformers/meters/feeders need attention
this week, and which ones.

Run from anywhere:
    python3 dashboard/app.py
Then open http://127.0.0.1:5000
"""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template, request, send_from_directory, session

import chatbot
import knowledge_base
import report

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

app = Flask(__name__)
# Only signs the session cookie holding chat follow-up context (last id/entity
# asked about); regenerated per process, so restarting the app just resets
# any open conversations rather than needing a stable stored secret.
app.secret_key = os.urandom(24)

# Status palette (good -> warning -> serious -> critical), fixed order,
# never reused for anything else on the page.
STATUS_COLORS = ["#0ca30c", "#fab219", "#ec835a", "#d03b3b"]

# Per-entity accent colors (blue/orange/aqua) live in static/style.css,
# keyed off each element's data-panel attribute rather than passed through
# here - avoids depending on CSS custom-property resolution for something
# only ever used for nav/header identity, never tier severity.

CHART_FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color="#0b0b0b")


def tier_chart(counts, tier_order, entity_label):
    total = counts.values.sum()
    pct = (counts.values / total * 100) if total else counts.values * 0
    fig = go.Figure(
        go.Bar(
            x=[t.title() for t in tier_order],
            y=counts.values,
            marker=dict(color=STATUS_COLORS[: len(tier_order)], cornerradius=6),
            text=counts.values,
            textposition="outside",
            cliponaxis=False,  # otherwise the tallest bar's label gets cut off by the plot area
            customdata=pct,
            hovertemplate=f"<b>%{{x}}</b><br>%{{y}} {entity_label} · %{{customdata:.1f}}%<extra></extra>",
            width=0.55,
        )
    )
    fig.update_layout(
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font=CHART_FONT,
        hoverlabel=dict(bgcolor="#232220", font_color="#fff", bordercolor="#232220"),
        xaxis=dict(showgrid=False, linecolor="#c3c2b7"),
        yaxis=dict(
            showgrid=True, gridcolor="#e1e0d9", zeroline=False, title=None,
            range=[0, max(counts.values) * 1.18],  # headroom for the outside label
        ),
        bargap=0.35,
        margin=dict(l=40, r=20, t=36, b=40),
        height=260,
        showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def landscape_chart(df, x_col, y_col, z_col, tier_col, id_col, labels, tier_order):
    """Rotatable 3D scatter of the real scored fleet, colored by the same
    fixed severity ramp as the tier bar chart - one more view of the same
    prediction, not a second unrelated visual."""
    sample = df if len(df) <= 1200 else df.sample(n=1200, random_state=42)
    fig = go.Figure()
    for tier, color in zip(tier_order, STATUS_COLORS):
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
            xaxis=dict(title=labels[0], backgroundcolor="#fcfcfb", gridcolor="#e1e0d9"),
            yaxis=dict(title=labels[1], backgroundcolor="#fcfcfb", gridcolor="#e1e0d9"),
            zaxis=dict(title=labels[2], backgroundcolor="#fcfcfb", gridcolor="#e1e0d9"),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
    )
    html = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    caption = (
        f"Showing a random sample of 1,200 of {len(df):,} for a responsive plot - drag to rotate."
        if len(df) > 1200 else "Drag to rotate, scroll to zoom."
    )
    return html, caption


def build_panel(
    csv_path, id_col, score_col, score_label, tier_col, flag_col, tier_order, entity_label, slug,
    raw_path=None, landscape_cols=None, landscape_labels=None,
):
    df = pd.read_csv(csv_path)
    counts = df[tier_col].value_counts().reindex(tier_order, fill_value=0)

    top = df.sort_values(score_col, ascending=False).head(15)
    has_reasons = "top_reasons" in df.columns
    top_rows = [
        {
            "id": row[id_col],
            "score": row[score_col],
            "tier": row[tier_col],
            "reasons": row["top_reasons"] if has_reasons else None,
        }
        for _, row in top.iterrows()
    ]

    landscape_html, landscape_caption = None, None
    if raw_path is not None:
        raw = pd.read_csv(raw_path)
        x_col, y_col = landscape_cols
        merged = raw.merge(df[[id_col, score_col, tier_col]], on=id_col)
        landscape_html, landscape_caption = landscape_chart(
            merged, x_col, y_col, score_col, tier_col, id_col, landscape_labels, tier_order
        )

    return {
        "slug": slug,
        "entity_label": entity_label,
        "total": f"{len(df):,}",
        "flagged": f"{int(df[flag_col].sum()):,}",
        "top_tier_count": f"{int(counts.iloc[-1]):,}",
        "top_tier_name": tier_order[-1].title(),
        "avg_score": round(df[score_col].mean(), 1),
        "score_label": score_label,
        "chart_html": tier_chart(counts, tier_order, entity_label),
        "top_rows": top_rows,
        "has_reasons": has_reasons,
        "tier_color_map": dict(zip(tier_order, STATUS_COLORS)),
        "landscape_html": landscape_html,
        "landscape_caption": landscape_caption,
    }


@app.route("/")
def index():
    transformer = build_panel(
        DATA / "transformer_risk_scores.csv",
        id_col="transformer_id",
        score_col="risk_score",
        score_label="Risk score",
        tier_col="risk_tier",
        flag_col="alert_flag",
        tier_order=["low", "moderate", "elevated", "critical"],
        entity_label="transformers",
        slug="transformer",
        raw_path=DATA / "transformer_data.csv",
        landscape_cols=("age_years", "temperature_rise_c"),
        landscape_labels=["Age (years)", "Temp rise (°C)", "Risk score"],
    )
    meter = build_panel(
        DATA / "meter_theft_scores.csv",
        id_col="meter_id",
        score_col="anomaly_score",
        score_label="Anomaly score",
        tier_col="priority_tier",
        flag_col="investigation_flag",
        tier_order=["low", "moderate", "elevated", "critical"],
        entity_label="meters",
        slug="meter",
        raw_path=DATA / "meter_data.csv",
        landscape_cols=("pct_drop_recent", "night_usage_ratio"),
        landscape_labels=["Recent usage drop (%)", "Night usage ratio", "Anomaly score"],
    )
    feeder = build_panel(
        DATA / "feeder_outage_scores.csv",
        id_col="feeder_id",
        score_col="outage_risk_score",
        score_label="Outage risk score",
        tier_col="risk_tier",
        flag_col="alert_flag",
        tier_order=["low", "moderate", "elevated", "critical"],
        entity_label="feeders",
        slug="feeder",
        raw_path=DATA / "feeder_data.csv",
        landscape_cols=("peak_load_pct", "vegetation_encroachment_score"),
        landscape_labels=["Peak load (%)", "Vegetation score", "Outage risk score"],
    )
    return render_template("index.html", transformer=transformer, meter=meter, feeder=feeder)


@app.route("/api/chat", methods=["POST"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "")
    context = session.get("chat_context", {})
    reply = chatbot.answer(message, DATA, context=context)
    session["chat_context"] = context

    focus_id = None
    last_id = context.get("last_id")
    if last_id and last_id[0] == "transformer":
        focus_id = chatbot.resolve_id("transformer", last_id[2], DATA)

    return jsonify({"reply": reply, "focus_id": focus_id})


@app.route("/api/chat/reset", methods=["POST"])
def chat_reset():
    session.pop("chat_context", None)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# AI Assistant tab APIs - the structured prediction card, history, compare,
# and PDF report all read the same raw+scores merge, an independent copy of
# the one in chatbot.py / report.py (see report.py's docstring for why: it
# avoids a circular import for one small join).
# --------------------------------------------------------------------------
def _load_transformer_full():
    scores = pd.read_csv(DATA / "transformer_risk_scores.csv")
    raw_cols = [
        "transformer_id", "age_years", "load_factor", "maintenance_score",
        "oil_quality_index", "temperature_rise_c", "location", "capacity_kva",
        "installation_year", "previous_failures", "last_serviced_date",
        "last_oil_replacement_date",
    ]
    raw = pd.read_csv(DATA / "transformer_data.csv")[raw_cols]
    return scores.merge(raw, on="transformer_id")


@app.route("/api/transformer/<transformer_id>")
def transformer_detail(transformer_id):
    full = _load_transformer_full()
    match = full[full["transformer_id"] == transformer_id]
    if match.empty:
        return jsonify({"error": f"{transformer_id} not found"}), 404
    row = match.iloc[0]
    tier = row["risk_tier"]
    pool = (
        full[full["transformer_id"] != transformer_id]
        .sort_values("risk_score", ascending=False)
        .head(30)["transformer_id"].tolist()
    )
    return jsonify({
        "id": row["transformer_id"],
        "risk_score": round(float(row["risk_score"]), 1),
        "risk_tier": tier,
        "health_score": round(float(row["health_score"]), 1),
        "confidence_pct": round(float(row["confidence_pct"]), 1),
        "predicted_failure_mode": row["predicted_failure_mode"],
        "recommendations": knowledge_base.MAINTENANCE_ACTIONS.get(tier, []),
        "compare_pool": pool,
    })


@app.route("/api/transformer/<transformer_id>/history")
def transformer_history(transformer_id):
    full = _load_transformer_full()
    match = full[full["transformer_id"] == transformer_id]
    if match.empty:
        return jsonify({"error": f"{transformer_id} not found"}), 404
    row = match.iloc[0]

    hist_all = pd.read_csv(DATA / "transformer_history.csv")
    hist = hist_all[hist_all["transformer_id"] == transformer_id].sort_values("month_offset")
    if hist.empty:
        return jsonify({"error": "no history data for this transformer"}), 404

    metrics = [
        ("oil_quality_index", "Oil quality index", "#2a78d6"),
        ("temperature_rise_c", "Temperature rise (°C)", "#e34948"),
        ("load_factor", "Load factor", "#1baf7a"),
    ]
    charts = []
    for metric, label, color in metrics:
        fig = go.Figure(
            go.Scatter(
                x=hist["month_offset"].tolist(), y=hist[metric].tolist(), mode="lines+markers",
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color, line=dict(width=2, color="#fcfcfb")),
                hovertemplate=f"month %{{x}}<br>{label}: %{{y:.2f}}<extra></extra>",
            )
        )
        fig.update_layout(
            height=180, margin=dict(l=10, r=10, t=24, b=24),
            paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=CHART_FONT, showlegend=False,
            title=dict(text=label, font=dict(size=12)),
            xaxis=dict(showgrid=False, title="Month"),
            yaxis=dict(showgrid=True, gridcolor="#e1e0d9", zeroline=False),
        )
        charts.append({"label": label, "figure": json.loads(fig.to_json())})

    return jsonify({
        "last_serviced": row["last_serviced_date"],
        "last_oil_replacement": row["last_oil_replacement_date"],
        "previous_failures": int(row["previous_failures"]),
        "charts": charts,
    })


@app.route("/api/transformer/<transformer_id>/compare/<other_id>")
def transformer_compare(transformer_id, other_id):
    full = _load_transformer_full()
    a_match = full[full["transformer_id"] == transformer_id]
    b_match = full[full["transformer_id"] == other_id]
    if a_match.empty or b_match.empty:
        return jsonify({"error": "transformer not found"}), 404
    a, b = a_match.iloc[0], b_match.iloc[0]

    metrics = [
        ("Risk score", "risk_score", "{:.1f}%"),
        ("Health score", "health_score", "{:.1f}%"),
        ("Confidence", "confidence_pct", "{:.1f}%"),
        ("Temperature rise", "temperature_rise_c", "{:.1f}°C"),
        ("Load factor", "load_factor", "{:.2f}"),
        ("Predicted failure", "predicted_failure_mode", "{}"),
    ]
    rows = [[label, fmt.format(a[col]), fmt.format(b[col])] for label, col, fmt in metrics]

    bar_metrics = ["risk_score", "health_score", "confidence_pct"]
    bar_labels = ["Risk score", "Health score", "Confidence"]
    fig = go.Figure([
        go.Bar(name=str(transformer_id), x=bar_labels, y=[float(a[m]) for m in bar_metrics],
               marker=dict(color="#2a78d6", cornerradius=6)),
        go.Bar(name=str(other_id), x=bar_labels, y=[float(b[m]) for m in bar_metrics],
               marker=dict(color="#eb6834", cornerradius=6)),
    ])
    fig.update_layout(
        barmode="group", height=260, margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=CHART_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, x=0),
        yaxis=dict(showgrid=True, gridcolor="#e1e0d9", range=[0, 100]),
    )

    return jsonify({"a": transformer_id, "b": other_id, "rows": rows, "figure": json.loads(fig.to_json())})


@app.route("/api/transformer/<transformer_id>/report", methods=["POST"])
def transformer_report(transformer_id):
    try:
        path = report.generate_pdf_report(transformer_id, DATA)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"filename": path.name, "url": f"/reports/{path.name}"})


@app.route("/reports/<path:filename>")
def reports(filename):
    # as_attachment=True: without it, send_from_directory serves the PDF
    # inline (Content-Disposition: inline), so a browser with a built-in PDF
    # viewer just opens it in the tab instead of downloading it - not what
    # the "Download <file>.pdf" link/button in the AI Assistant card implies.
    return send_from_directory(REPORTS, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
