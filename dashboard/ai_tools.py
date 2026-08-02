"""
SmartGrid PredictAI - LLM Assistant tools
=============================================
Each function here wraps existing, already-correct logic from chatbot.py /
report.py - the LLM's job is understanding the question and phrasing the
answer, it never invents a fact. Every number in every reply comes from one
of these tool calls reading the real scored CSVs, the same data the
rule-based chatbot.answer() (still used by the Streamlit console) reads.

Schemas are generated automatically from these functions' type hints and
docstrings by the @beta_tool decorator - see
https://platform.claude.com/docs (Tool Runner).
"""

import re
from pathlib import Path

from anthropic import beta_tool

import chatbot
import knowledge_base

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _num(transformer_id: str) -> int:
    """Extracts the numeric part of an id like "T0208" -> 208, the same
    zero-padding-agnostic lookup chatbot.py's rule-based engine already
    uses."""
    m = re.search(r"(\d+)$", transformer_id.strip())
    if not m:
        raise ValueError(f"'{transformer_id}' doesn't look like a transformer id (e.g. T0208)")
    return int(m.group(1))


@beta_tool
def predict_transformer(transformer_id: str) -> str:
    """Get the AI model's failure prediction for one transformer: risk score, risk tier, predicted failure mode, confidence, and the top recommended action.

    Args:
        transformer_id: The transformer's id, e.g. "T0208".
    """
    return chatbot._predict_reply(transformer_id, _num(transformer_id), DATA)


@beta_tool
def get_transformer_health(transformer_id: str) -> str:
    """Get a transformer's health score, service dates, remaining useful life estimate, and 12-month risk trend direction.

    Args:
        transformer_id: The transformer's id, e.g. "T0208".
    """
    return chatbot._health_reply(transformer_id, _num(transformer_id), DATA)


@beta_tool
def get_transformer_history(transformer_id: str) -> str:
    """Get a transformer's maintenance history: last service and oil-replacement dates, previous failure count, and the last 3 months of oil quality, temperature, and load readings.

    Args:
        transformer_id: The transformer's id, e.g. "T0208".
    """
    return chatbot._history_reply(transformer_id, _num(transformer_id), DATA)


@beta_tool
def get_transformer_details(transformer_id: str) -> str:
    """Get a transformer's asset details: location, capacity, installation year, current risk tier, and whether it's flagged for action.

    Args:
        transformer_id: The transformer's id, e.g. "T0208".
    """
    return chatbot._find_reply(transformer_id, _num(transformer_id), DATA)


@beta_tool
def get_transformer_trend(transformer_id: str) -> str:
    """Get whether a transformer's oil quality, temperature, and load have been trending up, down, or stable over the last 12 months.

    Args:
        transformer_id: The transformer's id, e.g. "T0208".
    """
    return chatbot._trend_reply(transformer_id, _num(transformer_id), DATA)


@beta_tool
def get_maintenance_recommendations(transformer_id: str) -> str:
    """Get the recommended maintenance actions for a transformer, based on its current risk tier.

    Args:
        transformer_id: The transformer's id, e.g. "T0208".
    """
    return chatbot._recommend_reply(transformer_id, _num(transformer_id), DATA)


@beta_tool
def compare_transformers(transformer_id_1: str, transformer_id_2: str) -> str:
    """Compare two transformers side by side: risk score, risk tier, health score, temperature, load factor, and predicted failure mode.

    Args:
        transformer_id_1: The first transformer's id, e.g. "T0208".
        transformer_id_2: The second transformer's id, e.g. "T0301".
    """
    return chatbot._compare_reply([_num(transformer_id_1), _num(transformer_id_2)], DATA)


@beta_tool
def generate_maintenance_report(transformer_id: str) -> str:
    """Generate and save a one-page PDF maintenance report for a transformer. Returns the saved file path.

    Args:
        transformer_id: The transformer's id, e.g. "T0208".
    """
    return chatbot._report_reply(transformer_id, _num(transformer_id), DATA)


@beta_tool
def query_fleet(
    entity: str,
    tier: str = "",
    high_risk_only: bool = False,
    flagged_only: bool = False,
    want_average: bool = False,
    want_reasons: bool = False,
    top_n: int = 10,
) -> str:
    """Query across the whole fleet of transformers, meters, or feeders: a count, a top-N list by risk score, an average score, or the most common reasons behind a tier's risk.

    Args:
        entity: Which asset type: "transformer", "meter", or "feeder".
        tier: Filter to one severity tier: "low", "moderate", "elevated", or "critical". Leave empty for no tier filter.
        high_risk_only: If true, filter to elevated-or-critical tier (overrides `tier`).
        flagged_only: If true, filter to only assets flagged for action.
        want_average: If true, return the average score for the filtered set instead of a list.
        want_reasons: If true, return the most common reasons behind the filtered set's risk - only meaningful together with a tier filter or high_risk_only.
        top_n: How many top-scoring assets to list. Ignored if want_average or want_reasons is true.
    """
    entity_key = entity.strip().lower().rstrip("s")
    if entity_key not in chatbot.ENTITIES:
        return f"Unknown entity '{entity}' - must be transformer, meter, or feeder."
    df, cfg = chatbot._load(entity_key, DATA)

    subset = df
    label_bits = [entity_key + "s"]
    if high_risk_only:
        subset = subset[subset[cfg["tier_col"]].isin(["elevated", "critical"])]
        label_bits.append("at high risk (elevated or critical)")
    elif tier:
        subset = subset[subset[cfg["tier_col"]] == tier.strip().lower()]
        label_bits.append(f"in the {tier.strip().lower()} tier")
    if flagged_only:
        subset = subset[subset[cfg["flag_col"]] == 1]
        label_bits.append("flagged for action")
    label = " ".join(label_bits)

    if want_reasons:
        if "top_reasons" not in df.columns or subset.empty:
            return f"No reason data for {label}."
        reasons = chatbot._common_reasons(subset)
        if not reasons:
            return f"No consistent reasons stand out for {label}."
        formatted = ", ".join(f"{phrase} ({count})" for phrase, count in reasons)
        return f"Most common reasons for {label} ({len(subset)} total): {formatted}."

    if want_average:
        if subset.empty:
            return f"No {label} to average."
        return f"Average {cfg['score_label']} for {label}: {subset[cfg['score_col']].mean():.1f}."

    if subset.empty:
        return f"No {label} right now."
    top = subset.sort_values(cfg["score_col"], ascending=False).head(max(1, min(top_n, 50)))
    ids = ", ".join(top[cfg["id_col"]].astype(str))
    remaining = len(subset) - len(top)
    more = f" (+{remaining} more)" if remaining > 0 else ""
    return f"{len(subset)} {label}, top {len(top)} by {cfg['score_label']}: {ids}{more}"


@beta_tool
def lookup_engineering_reference(query: str) -> str:
    """Look up a transformer engineering term (e.g. "dissolved gas analysis") or diagnose a described symptom (e.g. "loud humming noise") from the reference knowledge base.

    Args:
        query: The term to define, or the symptom to diagnose.
    """
    text = query.strip().lower()
    for term in sorted(knowledge_base.GLOSSARY, key=len, reverse=True):
        if term in text:
            return chatbot._glossary_reply(term)
    for symptom in sorted(knowledge_base.SYMPTOM_CAUSES, key=len, reverse=True):
        if symptom in text:
            return chatbot._diagnosis_reply(symptom)
    return f"No reference entry found for '{query}'."


ALL_TOOLS = [
    predict_transformer,
    get_transformer_health,
    get_transformer_history,
    get_transformer_details,
    get_transformer_trend,
    get_maintenance_recommendations,
    compare_transformers,
    generate_maintenance_report,
    query_fleet,
    lookup_engineering_reference,
]
