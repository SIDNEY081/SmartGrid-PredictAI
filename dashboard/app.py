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
#
# Split by domain, not by seniority: engineer owns asset reliability
# (transformer/feeder prediction + dispatching technicians to inspect
# them), investigator owns revenue protection (meter theft detection +
# dispatching technicians to investigate). Neither sees the other's
# dashboard - a role should never be able to view a panel it has no action
# on. Administrator sees and can act on everything, for oversight.
#
# dispatcher is a narrow role carved out of engineer's old day-to-day
# assignment duty: it sees only enough (transformer risk tiers, the asset
# management backlog) to do that one job, plus the assignments panel itself.
PANEL_ROLES = {
    "home": set(auth.ROLES),
    # manager gets read-only Q&A (predict/health/history/compare) but never
    # report generation - see REPORT_GENERATION_ROLES below and the
    # AI_SYSTEM_PROMPT rule that already tells the assistant to explain that
    # restriction rather than silently refuse.
    "assistant": {"administrator", "engineer", "manager"},
    "transformer": {"administrator", "engineer", "dispatcher", "manager"},
    "meter": {"administrator", "investigator", "manager"},
    "feeder": {"administrator", "engineer", "manager"},
    "asset_management": {"administrator", "engineer", "dispatcher", "manager"},
    "reports": {"administrator", "engineer"},
    "assignments": {"administrator", "engineer", "dispatcher"},
    "meter_assignments": {"administrator", "investigator"},
    "settings": {"administrator"},
    # Read-only activity/inspection/investigation trail - deliberately NOT
    # part of "settings" (which also manages user accounts), so an auditor
    # can review records without ever being able to touch a user.
    "audit": {"administrator", "auditor"},
}

# Roles allowed to assign/unassign technicians to transformers. Dispatcher is
# the primary, day-to-day owner of this queue, unrestricted by tier.
# Administrator retains it for oversight/setup. Engineer keeps it too, but
# only as an escalation fallback for when Dispatch hasn't gotten to a
# transformer yet - route-level access is granted here, then narrowed to
# overdue/emergency-only inside the handlers via
# auth.transformer_is_escalation_eligible().
ASSIGNMENT_ROLES = ("administrator", "dispatcher", "engineer")
ENGINEER_ESCALATION_ONLY = {"engineer"}

# Roles allowed to assign/unassign technicians to meters flagged for
# suspected theft - deliberately NOT engineer: revenue-protection dispatch
# is a different discipline from maintenance planning, handled by the
# investigator role (Revenue Protection Officer) instead.
METER_ASSIGNMENT_ROLES = ("administrator", "investigator")

# Valid statuses for a technician's theft-investigation submission - a
# different vocabulary from transformer inspections (pending/inspected/
# needs_followup) since the outcome of a theft investigation is "was there
# theft or not", not "does this need another visit".
METER_INVESTIGATION_STATUSES = ("pending", "investigating", "confirmed_theft", "false_positive")

# Roles allowed to generate a new PDF report - narrower than who can view the
# Reports panel or ask the AI Assistant questions: producing a report is a
# write action (a file on disk), viewing one isn't. Shared by the REST
# endpoint below and by the AI Assistant's tool filtering, so the chat
# interface can't be used as a side door around the same restriction.
REPORT_GENERATION_ROLES = ("administrator", "engineer")

# Status palette (good -> warning -> serious -> emergency), fixed order,
# never reused for anything else on the page. Matches style.css's
# --status-good/warning/serious/emergency tokens.
STATUS_COLORS = ["#22c55e", "#f2b53d", "#f2884b", "#ef4444"]

# Per-entity accent colors (blue/orange/aqua) live in static/style.css,
# keyed off each element's data-panel attribute rather than passed through
# here - avoids depending on CSS custom-property resolution for something
# only ever used for nav/header identity, never tier severity.

# Dark navy chart theme, matching style.css's --surface-1/--text-primary/
# --gridline/--border/--page-plane tokens. Plotly can't read CSS custom
# properties, so these are kept in sync by hand.
CHART_BG = "#132747"
CHART_HOVER_BG = "#0a1730"
CHART_GRID = "rgba(255, 255, 255, 0.10)"
CHART_AXIS_LINE = "rgba(255, 255, 255, 0.28)"
CHART_FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color="#eef3fc")


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
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        hoverlabel=dict(bgcolor=CHART_HOVER_BG, font_color="#eef3fc", bordercolor=CHART_AXIS_LINE),
        xaxis=dict(showgrid=False, linecolor=CHART_AXIS_LINE),
        yaxis=dict(
            showgrid=True, gridcolor=CHART_GRID, zeroline=False, title=None,
            range=[0, max(counts.values) * 1.18],  # headroom for the outside label
        ),
        bargap=0.35,
        margin=dict(l=40, r=20, t=36, b=40),
        autosize=True,
        showlegend=False,
    )
    # No fixed height - fills chart-wrap's stretched grid-row height (see
    # .content-grid / .chart-wrap in style.css) instead of leaving blank
    # space when the table next to it (e.g. the feeder-grouped transformer
    # worklist) is taller than a fixed 260px chart would be.
    return fig.to_html(
        full_html=False, include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )


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
        paper_bgcolor=CHART_BG,
        font=CHART_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1, x=0),
        scene=dict(
            xaxis=dict(title=labels[0], backgroundcolor=CHART_BG, gridcolor=CHART_GRID),
            yaxis=dict(title=labels[1], backgroundcolor=CHART_BG, gridcolor=CHART_GRID),
            zaxis=dict(title=labels[2], backgroundcolor=CHART_BG, gridcolor=CHART_GRID),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
    )
    html = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
    caption = (
        f"Showing a random sample of 1,200 of {len(df):,} for a responsive plot - drag to rotate."
        if len(df) > 1200 else "Drag to rotate, scroll to zoom."
    )
    return html, caption


def risk_map(df, lat_col, lon_col, tier_col, id_col, tier_order, entity_label):
    """Real geographic map of the scored fleet - actual gps_lat/gps_lon from
    the asset data, colored by the same severity ramp as the tier bar chart.
    Uses Carto's dark-matter basemap tiles, which (unlike Mapbox's own
    styles) render without an API token - appropriate for a prototype with
    no Mapbox account, and it matches the dark theme for free."""
    fig = go.Figure()
    for tier, color in zip(tier_order, STATUS_COLORS):
        sub = df[df[tier_col] == tier]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattermapbox(
                lat=sub[lat_col], lon=sub[lon_col],
                mode="markers",
                name=tier.title(),
                marker=dict(size=9, color=color, opacity=0.85),
                customdata=sub[[id_col]],
                hovertemplate=f"<b>%{{customdata[0]}}</b><extra></extra>",
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=CHART_BG,
        font=CHART_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"),
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=float(df[lat_col].mean()), lon=float(df[lon_col].mean())),
            zoom=9,
        ),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def theft_ntl_scatter(df, tariff):
    """Non-technical-loss scatter: billed revenue (from the customer-declared
    reading, actual_kwh) against metered consumption (the transformer feed
    estimate, expected_kwh) - both real per-meter fields already computed by
    models/theft_detection.py, not a synthetic demo axis. A theft account
    clusters low-right: the feed shows load being drawn but the bill doesn't
    reflect it. Flagged accounts (investigation_flag) are the same ones
    driving the meter panel's tier chart and worklist - one consistent flag,
    not a second detector."""
    normal = df[df["investigation_flag"] == 0]
    flagged = df[df["investigation_flag"] == 1]
    if len(normal) > 1500:
        normal = normal.sample(n=1500, random_state=42)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["expected_kwh"], y=normal["actual_kwh"] * tariff,
        mode="markers", name="Normal accounts",
        marker=dict(size=6, color="#5b8def", opacity=0.5, line=dict(width=0)),
        customdata=normal[["meter_id"]],
        hovertemplate="<b>%{customdata[0]}</b><br>Consumption: %{x:,.0f} kWh<br>Billed: R%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=flagged["expected_kwh"], y=flagged["actual_kwh"] * tariff,
        mode="markers", name="Flagged (high use, low bill)",
        marker=dict(size=8, color="#ef4444", opacity=0.9, line=dict(width=1, color="#fca5a5")),
        customdata=flagged[["meter_id"]],
        hovertemplate="<b>%{customdata[0]}</b><br>Consumption: %{x:,.0f} kWh<br>Billed: R%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=360,
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=CHART_FONT,
        hoverlabel=dict(bgcolor=CHART_HOVER_BG, font_color="#eef3fc", bordercolor=CHART_AXIS_LINE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Metered consumption (kWh)", showgrid=True, gridcolor=CHART_GRID, linecolor=CHART_AXIS_LINE),
        yaxis=dict(title="Billed revenue (R)", showgrid=True, gridcolor=CHART_GRID, linecolor=CHART_AXIS_LINE, zeroline=False),
        margin=dict(l=55, r=20, t=10, b=45),
        autosize=True,
    )
    return fig.to_html(
        full_html=False, include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )


def build_panel(
    csv_path, id_col, score_col, score_label, tier_col, flag_col, tier_order, entity_label, slug,
    raw_path=None, landscape_cols=None, landscape_labels=None, feeder_col=None, map_cols=None,
):
    df = pd.read_csv(csv_path)
    counts = df[tier_col].value_counts().reindex(tier_order, fill_value=0)

    top = df.sort_values(score_col, ascending=False).head(15)
    has_reasons = "top_reasons" in df.columns

    def _row(row):
        return {
            "id": row[id_col],
            "score": row[score_col],
            "tier": row[tier_col],
            "reasons": row["top_reasons"] if has_reasons else None,
        }

    top_rows = [_row(row) for _, row in top.iterrows()]

    # Grouped by feeder instead of a flat ranked list - a field crew plans a
    # route by feeder, not by an id-order that jumps across the whole
    # service area. Only the flagged (already-actionable) subset, worst
    # feeder first, worst transformer first within each feeder.
    feeder_groups = None
    if feeder_col is not None:
        flagged_df = df[df[flag_col] == 1].sort_values(score_col, ascending=False)
        groups = []
        for feeder_id, group in flagged_df.groupby(feeder_col, sort=False):
            groups.append({
                "feeder_id": feeder_id,
                "max_score": float(group[score_col].max()),
                "rows": [_row(row) for _, row in group.iterrows()],
            })
        feeder_groups = sorted(groups, key=lambda g: g["max_score"], reverse=True)

    landscape_html, landscape_caption = None, None
    map_html = None
    if raw_path is not None:
        raw = pd.read_csv(raw_path)
        x_col, y_col = landscape_cols
        merged = raw.merge(df[[id_col, score_col, tier_col]], on=id_col)
        landscape_html, landscape_caption = landscape_chart(
            merged, x_col, y_col, score_col, tier_col, id_col, landscape_labels, tier_order
        )
        if map_cols is not None:
            lat_col, lon_col = map_cols
            map_html = risk_map(merged, lat_col, lon_col, tier_col, id_col, tier_order, entity_label)

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
        "feeder_groups": feeder_groups,
        "has_reasons": has_reasons,
        "tier_color_map": dict(zip(tier_order, STATUS_COLORS)),
        "landscape_html": landscape_html,
        "landscape_caption": landscape_caption,
        "landscape_labels": landscape_labels,
        "map_html": map_html,
        "tier_order": tier_order,
    }


# Mirrors models/theft_detection.py's TARIFF_RAND_PER_KWH - kept as a
# separate constant rather than importing the models package here (this app
# deliberately never imports sklearn-dependent model code, only reads their
# CSV/joblib output), so bump both if the placeholder tariff ever changes.
TARIFF_RAND_PER_KWH_DISPLAY = 2.60


def build_executive_summary():
    """One cross-fleet summary for the Executive Overview - real counts
    pulled fresh from the same scored CSVs the per-entity panels use, not a
    second copy of the numbers. Returns plain ints/floats (not build_panel's
    comma-formatted strings) since these get summed across entity types."""
    transformers = pd.read_csv(DATA / "transformer_risk_scores.csv") if (DATA / "transformer_risk_scores.csv").exists() else None
    meters = pd.read_csv(DATA / "meter_theft_scores.csv") if (DATA / "meter_theft_scores.csv").exists() else None
    feeders = pd.read_csv(DATA / "feeder_outage_scores.csv") if (DATA / "feeder_outage_scores.csv").exists() else None

    total_assets = sum(len(df) for df in (transformers, meters, feeders) if df is not None)

    at_risk = 0
    for df, tier_col in ((transformers, "risk_tier"), (meters, "priority_tier"), (feeders, "risk_tier")):
        if df is not None:
            at_risk += int(df[tier_col].isin(["elevated", "emergency"]).sum())

    predicted_failures = int(transformers["alert_flag"].sum()) if transformers is not None else 0
    theft_alerts = int(meters["investigation_flag"].sum()) if meters is not None else 0
    outages_likely = int(feeders["alert_flag"].sum()) if feeders is not None else 0

    estimated_loss_rand = 0
    if meters is not None and "estimated_monthly_loss_rand" in meters.columns:
        actionable = meters[meters["priority_tier"].isin(["elevated", "emergency"])]
        estimated_loss_rand = int(actionable["estimated_monthly_loss_rand"].sum())

    return {
        "total_assets": f"{total_assets:,}",
        "at_risk": f"{at_risk:,}",
        "predicted_failures": f"{predicted_failures:,}",
        "theft_alerts": f"{theft_alerts:,}",
        "outages_likely": f"{outages_likely:,}",
        "estimated_loss_rand": f"R{estimated_loss_rand:,.0f}",
        "tariff": TARIFF_RAND_PER_KWH_DISPLAY,
    }


def build_asset_management():
    """Fleet-wide asset health rollup - transformer health/age/maintenance
    backlog plus a substation health ranking, none of which exist as a
    per-transformer detail field. Reuses _load_transformer_full() (defined
    below) rather than re-merging the two CSVs a second time."""
    full = _load_transformer_full()
    overdue = auth.count_overdue_transformer_maintenance()
    at_risk = int(full["risk_tier"].isin(["elevated", "emergency"]).sum())

    worst_health = full.sort_values("health_score").head(15)
    risk_rows = [
        {
            "id": row["transformer_id"],
            "substation": row["substation_name"],
            "health_score": round(float(row["health_score"]), 1),
            "age_years": round(float(row["age_years"]), 1),
            "risk_tier": row["risk_tier"],
            "next_maintenance_date": row.get("next_maintenance_date"),
        }
        for _, row in worst_health.iterrows()
    ]

    substations = (
        full.groupby(["substation_id", "substation_name", "cnc"])
        .agg(
            transformer_count=("transformer_id", "count"),
            avg_health_score=("health_score", "mean"),
            avg_age_years=("age_years", "mean"),
            at_risk_count=("risk_tier", lambda s: int(s.isin(["elevated", "emergency"]).sum())),
        )
        .reset_index()
        .sort_values("avg_health_score")
    )
    substation_rows = [
        {
            "substation_id": row["substation_id"],
            "substation_name": row["substation_name"],
            "cnc": row["cnc"],
            "transformer_count": int(row["transformer_count"]),
            "avg_health_score": round(float(row["avg_health_score"]), 1),
            "avg_age_years": round(float(row["avg_age_years"]), 1),
            "at_risk_count": int(row["at_risk_count"]),
        }
        for _, row in substations.iterrows()
    ]

    return {
        "total": f"{len(full):,}",
        "avg_health_score": round(float(full["health_score"].mean()), 1),
        "avg_age_years": round(float(full["age_years"].mean()), 1),
        "overdue": overdue,
        "at_risk": at_risk,
        "risk_rows": risk_rows,
        "substation_rows": substation_rows,
        "tier_color_map": dict(zip(["low", "moderate", "elevated", "emergency"], STATUS_COLORS)),
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
            tier_order=["low", "moderate", "elevated", "emergency"],
            entity_label="transformers",
            slug="transformer",
            raw_path=DATA / "transformer_data.csv",
            landscape_cols=("age_years", "temperature_rise_c"),
            landscape_labels=["Age (years)", "Temp rise (°C)", "Risk score"],
            feeder_col="feeder_id",
            map_cols=("gps_lat", "gps_lon"),
        )
    if role in PANEL_ROLES["meter"]:
        meter = build_panel(
            DATA / "meter_theft_scores.csv",
            id_col="meter_id",
            score_col="theft_risk_pct",
            score_label="Theft risk %",
            tier_col="priority_tier",
            flag_col="investigation_flag",
            tier_order=["low", "moderate", "elevated", "emergency"],
            entity_label="meters",
            slug="meter",
            raw_path=DATA / "meter_data.csv",
            landscape_cols=("pct_drop_recent", "night_usage_ratio"),
            landscape_labels=["Recent usage drop (%)", "Night usage ratio", "Theft risk %"],
            feeder_col="feeder_id",
            map_cols=("meter_lat", "meter_lon"),
        )
        meter["ntl_chart_html"] = theft_ntl_scatter(
            pd.read_csv(DATA / "meter_theft_scores.csv"), TARIFF_RAND_PER_KWH_DISPLAY
        )
    if role in PANEL_ROLES["feeder"]:
        feeder = build_panel(
            DATA / "feeder_outage_scores.csv",
            id_col="feeder_id",
            score_col="outage_risk_score",
            score_label="Outage risk score",
            tier_col="risk_tier",
            flag_col="alert_flag",
            tier_order=["low", "moderate", "elevated", "emergency"],
            entity_label="feeders",
            slug="feeder",
            raw_path=DATA / "feeder_data.csv",
            landscape_cols=("peak_load_pct", "vegetation_encroachment_score"),
            landscape_labels=["Peak load (%)", "Vegetation score", "Outage risk score"],
        )

    asset_management = None
    if role in PANEL_ROLES["asset_management"]:
        asset_management = build_asset_management()

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
    elif role == "auditor":
        home = {"kind": "auditor", "recent_activity_count": len(auth.recent_activity(limit=100))}
    else:
        home = {"kind": "fleet"}

    executive = build_executive_summary() if home["kind"] in ("administrator", "fleet") else None

    audit = None
    if role in PANEL_ROLES["audit"]:
        audit = {
            "activity": auth.recent_activity(limit=50),
            "inspections": auth.recent_inspections_fleet(limit=30),
            "meter_investigations": auth.recent_meter_investigations_fleet(limit=30),
        }

    # Real backlog counts, not a decorative widget - only computed for the
    # domain the signed-in role actually acts on (transformer dispatch for
    # engineers, meter dispatch for investigators). Formatted here rather
    # than in the template so plural/severity logic lives in one place.
    maintenance_queue = None
    if home["kind"] == "fleet":
        maintenance_queue = []
        if transformer:
            pending = auth.count_pending_transformer_inspections()
            maintenance_queue.append({
                "text": f"{pending} transformer inspection{'s' if pending != 1 else ''} pending",
                "severity": "warning" if pending else "good",
            })
            gap = auth.count_unassigned_emergency_transformers()
            if gap:
                maintenance_queue.append({
                    "text": f"{gap} emergency-tier transformer{'s' if gap != 1 else ''} not yet assigned to a technician",
                    "severity": "emergency",
                })
            if asset_management and asset_management["overdue"]:
                overdue = asset_management["overdue"]
                maintenance_queue.append({
                    "text": f"{overdue} transformer{'s' if overdue != 1 else ''} overdue for scheduled maintenance",
                    "severity": "emergency",
                })
        if meter:
            pending = auth.count_pending_meter_investigations()
            maintenance_queue.append({
                "text": f"{pending} meter investigation{'s' if pending != 1 else ''} pending",
                "severity": "warning" if pending else "good",
            })
            gap = auth.count_unassigned_emergency_meters()
            if gap:
                maintenance_queue.append({
                    "text": f"{gap} emergency-tier meter{'s' if gap != 1 else ''} not yet assigned for investigation",
                    "severity": "emergency",
                })

    return render_template(
        "index.html",
        user=user,
        role_label=auth.ROLE_LABELS[role],
        allowed_panels=allowed_panels,
        home=home,
        executive=executive,
        audit=audit,
        maintenance_queue=maintenance_queue,
        transformer=transformer, meter=meter, feeder=feeder,
        asset_management=asset_management,
    )


AI_SYSTEM_PROMPT = """You are Sidney, the AI assistant embedded in SmartGrid PredictAI - a \
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


def _load_meter_full():
    scores = pd.read_csv(DATA / "meter_theft_scores.csv")
    raw_cols = [
        "meter_id", "declared_kwh", "transformer_feed_estimate_kwh",
        "historical_avg_kwh", "pct_drop_recent", "night_usage_ratio",
        "meter_reversal_events_6mo", "zero_consumption_days_90d", "tamper_alarm_count",
    ]
    raw = pd.read_csv(DATA / "meter_data.csv")[raw_cols]
    return scores.merge(raw, on="meter_id")


@app.route("/api/meter/<meter_id>")
@auth.roles_required(*PANEL_ROLES["meter"])
def meter_detail(meter_id):
    full = _load_meter_full()
    match = full[full["meter_id"] == meter_id]
    if match.empty:
        return jsonify({"error": f"{meter_id} not found"}), 404
    row = match.iloc[0]
    tier = row["priority_tier"]
    history = auth.get_meter_investigation_history(meter_id)
    return jsonify({
        "id": row["meter_id"],
        "theft_risk_pct": round(float(row["theft_risk_pct"]), 1),
        "anomaly_score": round(float(row["anomaly_score"]), 1),
        "priority_tier": tier,
        "expected_kwh": round(float(row["expected_kwh"]), 1),
        "actual_kwh": round(float(row["actual_kwh"]), 1),
        "consumption_deviation_pct": round(float(row["consumption_deviation_pct"]), 1),
        "feeder_id": row.get("feeder_id"),
        "cnc": row.get("cnc"),
        "substation_id": row.get("substation_id"),
        "substation_name": row.get("substation_name"),
        "confirmed_incidents_nearby_12mo": int(row["confirmed_incidents_nearby_12mo"]),
        "top_reasons": row["top_reasons"] if "top_reasons" in row and pd.notna(row["top_reasons"]) else None,
        "recommendations": knowledge_base.THEFT_ACTIONS.get(tier, []),
        "investigation_history": history,
    })


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
                marker=dict(size=6, color=color, line=dict(width=2, color=CHART_BG)),
                hovertemplate=f"month %{{x}}<br>{label}: %{{y:.2f}}<extra></extra>",
            )
        )
        fig.update_layout(
            height=180, margin=dict(l=10, r=10, t=24, b=24),
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=CHART_FONT, showlegend=False,
            title=dict(text=label, font=dict(size=12)),
            xaxis=dict(showgrid=False, title="Month"),
            yaxis=dict(showgrid=True, gridcolor=CHART_GRID, zeroline=False),
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
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=CHART_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, x=0),
        yaxis=dict(showgrid=True, gridcolor=CHART_GRID, range=[0, 100]),
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
@auth.roles_required(*ASSIGNMENT_ROLES)
def admin_technicians():
    return jsonify({"technicians": auth.list_technicians()})


@app.route("/api/admin/technicians/<int:technician_id>/assignments", methods=["GET", "POST"])
@auth.roles_required(*ASSIGNMENT_ROLES)
def admin_technician_assignments(technician_id):
    if request.method == "POST":
        transformer_id = (request.get_json(silent=True) or {}).get("transformer_id", "").strip()
        valid_ids = set(pd.read_csv(DATA / "transformer_data.csv")["transformer_id"])
        if transformer_id not in valid_ids:
            return jsonify({"error": f"{transformer_id} is not a known transformer id"}), 400
        if auth.current_user()["role"] in ENGINEER_ESCALATION_ONLY and not auth.transformer_is_escalation_eligible(transformer_id):
            return jsonify({
                "error": f"{transformer_id} is not overdue or emergency-tier - routine assignments go through Dispatch."
            }), 403
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
@auth.roles_required(*ASSIGNMENT_ROLES)
def admin_technician_unassign(technician_id, transformer_id):
    if auth.current_user()["role"] in ENGINEER_ESCALATION_ONLY and not auth.transformer_is_escalation_eligible(transformer_id):
        return jsonify({
            "error": f"{transformer_id} is not overdue or emergency-tier - routine assignments go through Dispatch."
        }), 403
    auth.unassign_transformer(technician_id, transformer_id)
    auth.log_activity(auth.current_user(), f"unassigned {transformer_id} from technician #{technician_id}")
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Investigator (Revenue Protection) APIs - assign technicians to meters
# flagged for suspected theft. Same shape as the transformer assignment
# APIs above, gated on METER_ASSIGNMENT_ROLES instead of ASSIGNMENT_ROLES
# so an engineer can plan transformer maintenance but not theft dispatch.
# --------------------------------------------------------------------------
@app.route("/api/meter-technicians")
@auth.roles_required(*METER_ASSIGNMENT_ROLES)
def meter_technicians():
    return jsonify({"technicians": auth.list_technicians()})


@app.route("/api/meter-technicians/<int:technician_id>/assignments", methods=["GET", "POST"])
@auth.roles_required(*METER_ASSIGNMENT_ROLES)
def meter_technician_assignments(technician_id):
    if request.method == "POST":
        meter_id = (request.get_json(silent=True) or {}).get("meter_id", "").strip()
        valid_ids = set(pd.read_csv(DATA / "meter_data.csv")["meter_id"])
        if meter_id not in valid_ids:
            return jsonify({"error": f"{meter_id} is not a known meter id"}), 400
        auth.assign_meter(technician_id, meter_id)
        auth.log_activity(auth.current_user(), f"assigned meter {meter_id} to technician #{technician_id}")
        return jsonify({"ok": True}), 201

    assigned_ids = auth.get_assigned_meter_ids(technician_id)
    full = _load_meter_full()
    rows = full[full["meter_id"].isin(assigned_ids)]
    items = [
        {
            "id": row["meter_id"],
            "anomaly_score": round(float(row["anomaly_score"]), 1),
            "theft_risk_pct": round(float(row["theft_risk_pct"]), 1),
            "priority_tier": row["priority_tier"],
            "substation_name": row.get("substation_name"),
        }
        for _, row in rows.iterrows()
    ]
    return jsonify({"assignments": items})


@app.route("/api/meter-technicians/<int:technician_id>/assignments/<meter_id>", methods=["DELETE"])
@auth.roles_required(*METER_ASSIGNMENT_ROLES)
def meter_technician_unassign(technician_id, meter_id):
    auth.unassign_meter(technician_id, meter_id)
    auth.log_activity(auth.current_user(), f"unassigned meter {meter_id} from technician #{technician_id}")
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
            "feeder_id": row["feeder_id"],
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
# Technician APIs - assigned meters + theft investigation submissions.
# --------------------------------------------------------------------------
@app.route("/api/technician/meter-assignments")
@auth.roles_required("technician", "administrator")
def technician_meter_assignments():
    user = auth.current_user()
    technician_id = request.args.get("technician_id", type=int) if user["role"] == "administrator" else user["id"]
    if technician_id is None:
        return jsonify({"error": "technician_id is required for administrators"}), 400

    assigned_ids = auth.get_assigned_meter_ids(technician_id)
    full = _load_meter_full()
    rows = full[full["meter_id"].isin(assigned_ids)]

    items = []
    for _, row in rows.iterrows():
        history = auth.get_meter_investigation_history(row["meter_id"], limit=1)
        items.append({
            "id": row["meter_id"],
            "anomaly_score": round(float(row["anomaly_score"]), 1),
            "theft_risk_pct": round(float(row["theft_risk_pct"]), 1),
            "priority_tier": row["priority_tier"],
            "expected_kwh": round(float(row["expected_kwh"]), 1),
            "actual_kwh": round(float(row["actual_kwh"]), 1),
            "consumption_deviation_pct": round(float(row["consumption_deviation_pct"]), 1),
            "substation_name": row.get("substation_name"),
            "confirmed_incidents_nearby_12mo": int(row["confirmed_incidents_nearby_12mo"]),
            "recommended_action": row.get("recommended_action"),
            "top_reasons": row["top_reasons"] if "top_reasons" in row and pd.notna(row["top_reasons"]) else None,
            "last_investigation": history[0] if history else None,
        })
    return jsonify({"meters": items})


@app.route("/api/technician/recent-meter-investigations")
@auth.roles_required("technician", "administrator")
def technician_recent_meter_investigations():
    user = auth.current_user()
    technician_id = request.args.get("technician_id", type=int) if user["role"] == "administrator" else user["id"]
    if technician_id is None:
        return jsonify({"error": "technician_id is required for administrators"}), 400

    investigations = auth.get_recent_meter_investigations_by_technician(technician_id, days=90)
    return jsonify({"investigations": investigations})


@app.route("/api/technician/meter-investigation", methods=["POST"])
@auth.roles_required("technician", "administrator")
def technician_meter_investigation():
    user = auth.current_user()
    meter_id = request.form.get("meter_id", "").strip()
    status = request.form.get("status", "").strip()
    notes = request.form.get("notes", "").strip()

    if not meter_id or status not in METER_INVESTIGATION_STATUSES:
        return jsonify({"error": "meter_id and a valid status are required"}), 400

    photo_filename = None
    photo = request.files.get("photo")
    if photo and photo.filename:
        ext = photo.filename.rsplit(".", 1)[-1].lower() if "." in photo.filename else ""
        if ext not in ALLOWED_PHOTO_EXTENSIONS:
            return jsonify({"error": f"unsupported photo type: .{ext}"}), 400
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        photo_filename = secure_filename(f"{meter_id}_{stamp}_{photo.filename}")
        photo.save(auth.UPLOAD_DIR / photo_filename)

    auth.submit_meter_investigation(meter_id, user["id"], status, notes, photo_filename)
    auth.log_activity(user, f"submitted theft investigation for meter {meter_id} ({status})")
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
