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

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template, request

import chatbot

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

app = Flask(__name__)

# Status palette (good -> warning -> serious -> critical), fixed order,
# never reused for anything else on the page.
STATUS_COLORS = ["#0ca30c", "#fab219", "#ec835a", "#d03b3b"]

# Per-entity accent colors (blue/orange/aqua) live in static/style.css,
# keyed off each element's data-panel attribute rather than passed through
# here - avoids depending on CSS custom-property resolution for something
# only ever used for nav/header identity, never tier severity.

CHART_FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color="#0b0b0b")


def tier_chart(counts, tier_order, entity_label):
    fig = go.Figure(
        go.Bar(
            x=[t.title() for t in tier_order],
            y=counts.values,
            marker_color=STATUS_COLORS[: len(tier_order)],
            text=counts.values,
            textposition="outside",
            cliponaxis=False,  # otherwise the tallest bar's label gets cut off by the plot area
            hovertemplate=f"%{{x}}: %{{y}} {entity_label}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font=CHART_FONT,
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


def build_panel(csv_path, id_col, score_col, score_label, tier_col, flag_col, tier_order, entity_label, slug):
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
    )
    return render_template("index.html", transformer=transformer, meter=meter, feeder=feeder)


@app.route("/api/chat", methods=["POST"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "")
    reply = chatbot.answer(message, DATA)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
