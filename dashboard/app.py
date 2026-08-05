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
import re
import secrets
from datetime import datetime
from functools import wraps
from pathlib import Path

import anthropic
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

import ai_tools
import auth
import chatbot
import knowledge_base
import report

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

app = Flask(__name__)
# Signs both the login session cookie and the chat follow-up context (last
# id/entity asked about). Falls back to a per-process random key for local
# dev; set SECRET_KEY in production so restarts (e.g. a free-tier host
# spinning the dyno down after idle) don't invalidate every open session.
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
app.teardown_appcontext(auth.close_db)
auth.init_db()

# Imported only after init_db() - its module-level DataFrames (for the Power
# BI Desktop connector) run SQL queries against app.db at import time, which
# would fail with "no such table: users" on a fresh DB that hasn't been
# seeded yet (e.g. a brand new deploy).
import powerbi_data

# API key for the read-only /api/powerbi/* routes below - generated once and
# persisted to disk (not gitignored-secret-worthy, just a local prototype
# credential) so it survives app restarts and a Power BI Service connection
# keeps working instead of breaking every time the dev server reloads.
POWERBI_KEY_PATH = DATA / "powerbi_api_key.txt"
if POWERBI_KEY_PATH.exists():
    POWERBI_API_KEY = POWERBI_KEY_PATH.read_text().strip()
else:
    POWERBI_API_KEY = secrets.token_urlsafe(32)
    POWERBI_KEY_PATH.write_text(POWERBI_API_KEY)

# Roles allowed into each nav panel - enforced both server-side (route
# decorators below) and client-side (index.html only renders the nav
# buttons/panels a role can reach), so hiding a button is a UX nicety, not
# the actual access boundary.
PANEL_ROLES = {
    "home": set(auth.ROLES),
    "assistant": {"administrator", "engineer"},
    "transformer": {"administrator", "engineer"},
    "meter": {"administrator", "engineer"},
    "feeder": {"administrator", "engineer"},
    "reports": {"administrator", "engineer"},
    "settings": {"administrator"},
}

# Roles allowed to generate a new PDF report - narrower than who can view the
# Reports panel or ask the AI Assistant questions: producing a report is a
# write action (a file on disk), viewing one isn't. Shared by the REST
# endpoint below and by the AI Assistant's tool filtering, so the chat
# interface can't be used as a side door around the same restriction.
REPORT_GENERATION_ROLES = ("administrator", "engineer")

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
        "landscape_labels": landscape_labels,
        "tier_order": tier_order,
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = auth.verify_login(username, password)
        if user is None:
            error = "Incorrect username or password."
        else:
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]
            auth.log_activity(user, "logged in")
            next_path = request.form.get("next") or url_for("index")
            return redirect(next_path)

    return render_template(
        "login.html", error=error, role_labels=auth.ROLE_LABELS, next_path=request.args.get("next", ""),
        fleet=auth.fleet_counts(),
    )


@app.route("/logout", methods=["POST"])
def logout():
    user = auth.current_user()
    if user:
        auth.log_activity(user, "logged out")
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@auth.login_required
def index():
    user = auth.current_user()
    role = user["role"]
    allowed_panels = {panel for panel, roles in PANEL_ROLES.items() if role in roles}

    transformer = meter = feeder = None
    if role in PANEL_ROLES["transformer"]:
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
    if role in PANEL_ROLES["meter"]:
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
    if role in PANEL_ROLES["feeder"]:
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

    home = None
    if role == "administrator":
        home = {
            "kind": "administrator",
            "user_count": len(auth.list_users()),
            "dataset_stats": auth.dataset_stats(),
            "activity": auth.recent_activity(limit=15),
        }
    elif role == "technician":
        home = {"kind": "technician"}
    else:
        home = {"kind": "fleet"}

    return render_template(
        "index.html",
        user=user,
        role_label=auth.ROLE_LABELS[role],
        allowed_panels=allowed_panels,
        home=home,
        transformer=transformer, meter=meter, feeder=feeder,
    )


AI_SYSTEM_PROMPT = """You are PredictAI, the AI assistant embedded in SmartGrid PredictAI - a \
predictive-maintenance platform for electricity utility engineers, covering the \
transformer, meter, and feeder fleet around Tzaneen and the Mopani District. You \
help maintenance engineers and system administrators interpret failure predictions, \
theft detection, and outage forecasts.

You are a technical assistant for the people who run the grid, not a customer-support \
bot. Answer like a knowledgeable colleague: direct, specific, and grounded in the \
actual scored data.

Rules:
- Never state a risk score, health score, date, location, or any other fact about a \
specific transformer/meter/feeder without calling a tool first. You have no memory of \
the fleet's data - every number must come from a tool result.
- If a tool reports an asset doesn't exist, say so plainly - don't guess at a substitute id.
- Keep replies concise and skimmable - this is read in a chat panel, not a report.
- For general engineering questions (e.g. "what is dissolved gas analysis?", "what \
causes a humming noise?"), use lookup_engineering_reference rather than answering from \
general knowledge, so answers stay consistent with this platform's reference material.
- If a question is ambiguous about which asset it means, ask a brief clarifying \
question rather than guessing.
- If someone asks you to generate a PDF maintenance report and you don't have a tool \
for it, that's a permissions limit, not a missing feature - tell them report \
generation is restricted to engineers and administrators, and to ask one of them.
"""

_anthropic_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


@app.route("/api/chat", methods=["POST"])
@auth.roles_required(*PANEL_ROLES["assistant"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not message:
        return jsonify({"reply": "Ask me something about a transformer, meter, or feeder.", "focus_id": None})

    user = auth.current_user()
    history = auth.get_chat_history(user["id"], limit=20)
    history.append({"role": "user", "content": message})

    # Same restriction as the /report REST endpoint: only administrator/
    # engineer can generate a PDF, whether they ask for it by clicking a
    # button or by asking the chat assistant. Dropping the tool from the
    # list (rather than trusting the model to decline) means the model
    # can't be talked into calling it either.
    tools = ai_tools.ALL_TOOLS
    if user["role"] not in REPORT_GENERATION_ROLES:
        tools = [t for t in tools if t.name != "generate_maintenance_report"]

    try:
        runner = get_anthropic_client().beta.messages.tool_runner(
            model="claude-sonnet-5",
            max_tokens=1536,
            system=AI_SYSTEM_PROMPT,
            tools=tools,
            output_config={"effort": "medium"},
            messages=history,
        )
        tool_calls, final_message = [], None
        for msg in runner:
            final_message = msg
            tool_calls.extend(b for b in msg.content if b.type == "tool_use")
    except Exception as exc:
        # A missing ANTHROPIC_API_KEY surfaces as a plain TypeError from deep
        # inside the SDK's auth-header construction, not an AuthenticationError
        # (that class is only for a *rejected* key, i.e. a 401 response) - so
        # this checks both the typed exception and the message text rather
        # than relying on isinstance() alone.
        text = str(exc).lower()
        if isinstance(exc, anthropic.AuthenticationError) or "api_key" in text or "authentication" in text:
            return jsonify({
                "reply": "The AI Assistant isn't configured yet - ask an administrator to set "
                         "the ANTHROPIC_API_KEY environment variable.",
                "focus_id": None,
            }), 503
        return jsonify({"reply": f"Sorry, the AI Assistant hit an error: {exc}", "focus_id": None}), 502

    reply_text = "".join(
        b.text for b in (final_message.content if final_message else []) if b.type == "text"
    ).strip()
    if not reply_text:
        reply_text = "I wasn't able to put together an answer for that - try rephrasing?"

    auth.save_chat_message(user["id"], "user", message)
    auth.save_chat_message(user["id"], "assistant", reply_text)
    auth.log_activity(user, "asked the AI Assistant a question")

    # Focus id for the prediction card: whichever transformer the model
    # actually called a tool for, most recent call wins - resolved against
    # the real CSV so a malformed id from the model never reaches the UI.
    focus_id = None
    for call in reversed(tool_calls):
        inp = call.input or {}
        raw_id = inp.get("transformer_id_2") or inp.get("transformer_id_1") or inp.get("transformer_id")
        if raw_id:
            match = re.search(r"(\d+)$", str(raw_id))
            if match:
                focus_id = chatbot.resolve_id("transformer", int(match.group(1)), DATA)
            break

    return jsonify({"reply": reply_text, "focus_id": focus_id})


@app.route("/api/chat/reset", methods=["POST"])
@auth.roles_required(*PANEL_ROLES["assistant"])
def chat_reset():
    auth.clear_chat_history(auth.current_user()["id"])
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
        "transformer_id", "transformer_name", "cnc", "substation_id",
        "substation_name", "pole_id", "gps_lat", "gps_lon",
        "age_years", "load_factor", "maintenance_score",
        "oil_quality_index", "temperature_rise_c", "capacity_kva",
        "installation_year", "previous_failures", "last_serviced_date",
        "last_oil_replacement_date",
    ]
    raw = pd.read_csv(DATA / "transformer_data.csv")[raw_cols]
    return scores.merge(raw, on="transformer_id")


@app.route("/api/transformer/<transformer_id>")
@auth.roles_required(*PANEL_ROLES["transformer"])
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
@auth.roles_required(*PANEL_ROLES["transformer"])
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
@auth.roles_required(*PANEL_ROLES["transformer"])
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
@auth.roles_required(*REPORT_GENERATION_ROLES)
def transformer_report(transformer_id):
    try:
        path = report.generate_pdf_report(transformer_id, DATA)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    auth.log_activity(auth.current_user(), f"generated maintenance report for {transformer_id}")
    return jsonify({"filename": path.name, "url": f"/reports/{path.name}"})


@app.route("/api/reports")
@auth.roles_required(*PANEL_ROLES["reports"])
def list_reports():
    REPORTS.mkdir(parents=True, exist_ok=True)
    files = sorted(REPORTS.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify({
        "reports": [
            {
                "filename": f.name,
                "url": f"/reports/{f.name}",
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
            for f in files
        ]
    })


@app.route("/reports/<path:filename>")
@auth.login_required
def reports(filename):
    # as_attachment=True (the default here) forces a download; ?view=1
    # serves the same file inline (Content-Disposition: inline) so a
    # browser's built-in PDF viewer opens it in a new tab instead - the
    # "View" vs "Download" choice the UI now offers for the same file.
    inline = request.args.get("view") == "1"
    return send_from_directory(REPORTS, filename, as_attachment=not inline)


# --------------------------------------------------------------------------
# Administrator APIs - user management, activity log, dataset visibility.
# --------------------------------------------------------------------------
@app.route("/api/admin/users", methods=["GET", "POST"])
@auth.roles_required("administrator")
def admin_users():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        try:
            auth.create_user(
                payload.get("username", "").strip(),
                payload.get("password", ""),
                payload.get("full_name", "").strip(),
                payload.get("role", ""),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        auth.log_activity(auth.current_user(), f"created user '{payload.get('username')}' ({payload.get('role')})")
        return jsonify({"ok": True}), 201

    return jsonify({"users": auth.list_users(), "roles": auth.ROLES, "role_labels": auth.ROLE_LABELS})


@app.route("/api/admin/users/<int:user_id>/active", methods=["POST"])
@auth.roles_required("administrator")
def admin_set_user_active(user_id):
    payload = request.get_json(silent=True) or {}
    auth.set_user_active(user_id, bool(payload.get("is_active", True)))
    auth.log_activity(auth.current_user(), f"set user #{user_id} active={bool(payload.get('is_active', True))}")
    return jsonify({"ok": True})


@app.route("/api/admin/activity")
@auth.roles_required("administrator")
def admin_activity():
    return jsonify({"activity": auth.recent_activity(limit=50)})


@app.route("/api/admin/datasets")
@auth.roles_required("administrator")
def admin_datasets():
    return jsonify({"datasets": auth.dataset_stats()})


@app.route("/api/admin/technicians")
@auth.roles_required("administrator")
def admin_technicians():
    return jsonify({"technicians": auth.list_technicians()})


@app.route("/api/admin/technicians/<int:technician_id>/assignments", methods=["GET", "POST"])
@auth.roles_required("administrator")
def admin_technician_assignments(technician_id):
    if request.method == "POST":
        transformer_id = (request.get_json(silent=True) or {}).get("transformer_id", "").strip()
        valid_ids = set(pd.read_csv(DATA / "transformer_data.csv")["transformer_id"])
        if transformer_id not in valid_ids:
            return jsonify({"error": f"{transformer_id} is not a known transformer id"}), 400
        auth.assign_transformer(technician_id, transformer_id)
        auth.log_activity(auth.current_user(), f"assigned {transformer_id} to technician #{technician_id}")
        return jsonify({"ok": True}), 201

    assigned_ids = auth.get_assigned_transformer_ids(technician_id)
    full = _load_transformer_full()
    rows = full[full["transformer_id"].isin(assigned_ids)]
    items = [
        {"id": row["transformer_id"], "cnc": row["cnc"], "risk_tier": row["risk_tier"], "status": row["status"]}
        for _, row in rows.iterrows()
    ]
    return jsonify({"assignments": items})


@app.route("/api/admin/technicians/<int:technician_id>/assignments/<transformer_id>", methods=["DELETE"])
@auth.roles_required("administrator")
def admin_technician_unassign(technician_id, transformer_id):
    auth.unassign_transformer(technician_id, transformer_id)
    auth.log_activity(auth.current_user(), f"unassigned {transformer_id} from technician #{technician_id}")
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Technician APIs - assigned transformers + inspection submissions.
# --------------------------------------------------------------------------
ALLOWED_PHOTO_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


@app.route("/api/technician/assignments")
@auth.roles_required("technician", "administrator")
def technician_assignments():
    user = auth.current_user()
    technician_id = request.args.get("technician_id", type=int) if user["role"] == "administrator" else user["id"]
    if technician_id is None:
        return jsonify({"error": "technician_id is required for administrators"}), 400

    assigned_ids = auth.get_assigned_transformer_ids(technician_id)
    full = _load_transformer_full()
    rows = full[full["transformer_id"].isin(assigned_ids)]

    items = []
    for _, row in rows.iterrows():
        tier = row["risk_tier"]
        history = auth.get_inspection_history(row["transformer_id"], limit=1)
        items.append({
            "id": row["transformer_id"],
            "risk_score": round(float(row["risk_score"]), 1),
            "risk_tier": tier,
            "health_score": round(float(row["health_score"]), 1),
            "predicted_failure_mode": row["predicted_failure_mode"],
            "recommendations": knowledge_base.MAINTENANCE_ACTIONS.get(tier, []),
            "cnc": row["cnc"],
            "last_inspection": history[0] if history else None,
        })
    return jsonify({"transformers": items})


@app.route("/api/technician/recent-inspections")
@auth.roles_required("technician", "administrator")
def technician_recent_inspections():
    user = auth.current_user()
    technician_id = request.args.get("technician_id", type=int) if user["role"] == "administrator" else user["id"]
    if technician_id is None:
        return jsonify({"error": "technician_id is required for administrators"}), 400

    inspections = auth.get_recent_inspections_by_technician(technician_id, days=90)
    if not inspections:
        return jsonify({"inspections": []})

    full = _load_transformer_full()
    by_id = full.set_index("transformer_id")
    items = []
    for insp in inspections:
        row = by_id.loc[insp["transformer_id"]] if insp["transformer_id"] in by_id.index else None
        items.append({
            "transformer_id": insp["transformer_id"],
            "cnc": row["cnc"] if row is not None else None,
            "status": insp["status"],
            "notes": insp["notes"],
            "created_at": insp["created_at"],
            "technician_name": insp["technician_name"],
        })
    return jsonify({"inspections": items})


@app.route("/api/technician/inspection", methods=["POST"])
@auth.roles_required("technician", "administrator")
def technician_inspection():
    user = auth.current_user()
    transformer_id = request.form.get("transformer_id", "").strip()
    status = request.form.get("status", "").strip()
    notes = request.form.get("notes", "").strip()

    if not transformer_id or status not in ("pending", "inspected", "needs_followup"):
        return jsonify({"error": "transformer_id and a valid status are required"}), 400

    photo_filename = None
    photo = request.files.get("photo")
    if photo and photo.filename:
        ext = photo.filename.rsplit(".", 1)[-1].lower() if "." in photo.filename else ""
        if ext not in ALLOWED_PHOTO_EXTENSIONS:
            return jsonify({"error": f"unsupported photo type: .{ext}"}), 400
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        photo_filename = secure_filename(f"{transformer_id}_{stamp}_{photo.filename}")
        photo.save(auth.UPLOAD_DIR / photo_filename)

    auth.submit_inspection(transformer_id, user["id"], status, notes, photo_filename)
    auth.log_activity(user, f"submitted inspection for {transformer_id} ({status})")
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Power BI API - read-only JSON versions of powerbi_data.py's tables, for
# Power BI Service's scheduled refresh (its Web connector can't read local
# files, unlike Power BI Desktop's Python-script connector). Key-gated
# rather than session-gated: Power BI's Web connector authenticates with a
# static header/query param, not a login flow.
# --------------------------------------------------------------------------
def powerbi_key_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key != POWERBI_API_KEY:
            return jsonify({"error": "missing or invalid API key"}), 401
        return view(*args, **kwargs)

    return wrapped


@app.route("/api/powerbi/<table_name>")
@powerbi_key_required
def powerbi_table(table_name):
    builder = powerbi_data.TABLES.get(table_name)
    if builder is None:
        return jsonify({"error": f"unknown table '{table_name}'", "tables": sorted(powerbi_data.TABLES)}), 404
    df = builder()
    return app.response_class(df.to_json(orient="records", date_format="iso"), mimetype="application/json")


if __name__ == "__main__":
    print("=" * 60)
    print("Power BI API key (Web connector, header X-API-Key or ?api_key=):")
    print(f"  {POWERBI_API_KEY}")
    print(f"  Tables: {', '.join(sorted(powerbi_data.TABLES))}")
    print("=" * 60)
    app.run(debug=True, port=5000)
