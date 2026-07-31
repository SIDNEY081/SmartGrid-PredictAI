"""
SmartGrid PredictAI - PDF Maintenance Report
================================================
One-page PDF per transformer (prediction summary, top reasons, recommended
actions) built with reportlab - pure-Python, no system deps, so it works
the same on Windows as everywhere else (unlike weasyprint's Cairo/Pango
requirement).

Deliberately does NOT import dashboard/chatbot.py: chatbot.py imports this
module for its "generate report" intent, so importing chatbot.py back here
would create a circular import. The small raw+scores merge is duplicated
from chatbot._load_transformer_full instead of shared.
"""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import knowledge_base

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "reports"


def _load_transformer_row(transformer_id, data_dir):
    scores = pd.read_csv(data_dir / "transformer_risk_scores.csv")
    raw_cols = [
        "transformer_id", "location", "capacity_kva", "installation_year",
        "previous_failures", "last_serviced_date", "last_oil_replacement_date",
    ]
    raw = pd.read_csv(data_dir / "transformer_data.csv")[raw_cols]
    full = scores.merge(raw, on="transformer_id")
    match = full[full["transformer_id"] == transformer_id]
    return None if match.empty else match.iloc[0]


def generate_pdf_report(transformer_id, data_dir, out_dir=None):
    """Builds reports/<transformer_id>_report.pdf and returns its Path.
    Raises ValueError if transformer_id has no row in transformer_risk_scores.csv."""
    row = _load_transformer_row(transformer_id, data_dir)
    if row is None:
        raise ValueError(f"{transformer_id} not found in transformer_risk_scores.csv")

    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{transformer_id}_report.pdf"

    tier = row["risk_tier"]
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"SmartGrid PredictAI &ndash; Maintenance Report: {transformer_id}", styles["Title"]),
        Spacer(1, 10),
        Paragraph(
            "Generated from data/transformer_risk_scores.csv and data/transformer_data.csv "
            "(synthetic data prototype).",
            styles["Normal"],
        ),
        Spacer(1, 14),
    ]

    fields = [
        ("Risk Score", f"{row['risk_score']:.1f}%"),
        ("Risk Level", tier.capitalize()),
        ("Predicted Failure", row["predicted_failure_mode"]),
        ("Confidence", f"{row['confidence_pct']:.1f}%"),
        ("Health Score", f"{row['health_score']:.1f}/100"),
        ("Remaining Useful Life", f"{row['remaining_useful_life_years']:.1f} years"),
        ("Location", row["location"]),
        ("Capacity", f"{row['capacity_kva']} kVA"),
        ("Installed", str(int(row["installation_year"]))),
        ("Previous Failures", str(int(row["previous_failures"]))),
        ("Last Serviced", row["last_serviced_date"]),
        ("Last Oil Replacement", row["last_oil_replacement_date"]),
        ("Next Maintenance", row["next_maintenance_date"]),
    ]
    table = Table([["Field", "Value"]] + [[k, str(v)] for k, v in fields], colWidths=[170, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Top Reasons", styles["Heading2"]))
    story.append(Paragraph(str(row.get("top_reasons") or "n/a"), styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Recommended Actions", styles["Heading2"]))
    for action in knowledge_base.MAINTENANCE_ACTIONS.get(tier, []):
        story.append(Paragraph(f"&bull; {action}", styles["Normal"]))

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, title=f"{transformer_id} Maintenance Report")
    doc.build(story)
    return out_path
