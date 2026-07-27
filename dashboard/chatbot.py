"""
SmartGrid PredictAI - Dashboard Chat
=======================================
A small rule-based (not LLM-backed) query engine over the three score CSVs.
No external API, no network dependency, no cost - just keyword matching
against a question, mapped onto a pandas filter. Good enough for "how many
transformers are critical?" or "why is T0208 high risk?" style questions;
not a general chatbot. Shared by both dashboard/app.py (Flask) and
dashboard/streamlit_app.py (Streamlit) - the same rules answer either UI.
"""

import re
from collections import Counter

import pandas as pd

ENTITIES = {
    "transformer": {
        "csv": "transformer_risk_scores.csv",
        "id_col": "transformer_id",
        "id_prefix": "T",
        "score_col": "risk_score",
        "score_label": "risk score",
        "tier_col": "risk_tier",
        "flag_col": "alert_flag",
        "aliases": ["transformer", "transformers"],
    },
    "meter": {
        "csv": "meter_theft_scores.csv",
        "id_col": "meter_id",
        "id_prefix": "M",
        "score_col": "anomaly_score",
        "score_label": "anomaly score",
        "tier_col": "priority_tier",
        "flag_col": "investigation_flag",
        "aliases": ["meter", "meters"],
    },
    "feeder": {
        "csv": "feeder_outage_scores.csv",
        "id_col": "feeder_id",
        "id_prefix": "F",
        "score_col": "outage_risk_score",
        "score_label": "outage risk score",
        "tier_col": "risk_tier",
        "flag_col": "alert_flag",
        "aliases": ["feeder", "feeders"],
    },
}

TIERS = ["low", "moderate", "elevated", "critical"]
# Older/alternate wording someone might still say out loud even though the
# dashboard itself standardized on the four words above.
TIER_SYNONYMS = {
    "urgent": "critical", "high": "elevated", "medium": "moderate",
}

ID_PATTERN = re.compile(r"\b([TMF])0*(\d{1,5})\b", re.IGNORECASE)
TOP_N_PATTERN = re.compile(r"\btop\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:highest|riskiest|worst|most)\b")
GREETING_PATTERN = re.compile(r"^\s*(hi|hello|hey|howdy|yo)\b[!.\s]*$")
HELP_PATTERN = re.compile(r"\b(help|what can you (do|ask)|options|commands)\b")

HELP_TEXT = (
    "I can answer questions about transformers, meters, and feeders. Try things like:\n"
    "• “how many transformers are critical?”\n"
    "• “what's the risk score for T0208?”\n"
    "• “why is T0208 high risk?”\n"
    "• “why are feeders critical?” (most common reasons across a group)\n"
    "• “top 5 riskiest meters”\n"
    "• “average anomaly score for meters”\n"
    "• “which feeders are flagged?”"
)
GREETING_TEXT = "Hi! " + HELP_TEXT


def _load(entity_key, data_dir):
    cfg = ENTITIES[entity_key]
    return pd.read_csv(data_dir / cfg["csv"]), cfg


def _find_entity(text):
    for key, cfg in ENTITIES.items():
        for alias in cfg["aliases"]:
            if re.search(rf"\b{alias}\b", text):
                return key
    return None


def _find_tier(text):
    for tier in TIERS:
        if tier in text:
            return tier
    for synonym, tier in TIER_SYNONYMS.items():
        if synonym in text:
            return tier
    return None


def _find_id(raw_text):
    m = ID_PATTERN.search(raw_text)
    if not m:
        return None
    prefix, digits = m.group(1).upper(), int(m.group(2))
    for key, cfg in ENTITIES.items():
        if cfg["id_prefix"] == prefix:
            return key, prefix, digits
    return None


def _find_top_n(text, default=10, cap=50):
    m = TOP_N_PATTERN.search(text)
    if not m:
        return default
    n = int(m.group(1) or m.group(2))
    return max(1, min(n, cap))


def _common_reasons(subset, top_k=5):
    """Tally the individual reason phrases (e.g. "high age_years") across a
    subset's top_reasons column and return the most frequent ones, most
    common first."""
    counter = Counter()
    for cell in subset["top_reasons"].dropna():
        for phrase in str(cell).split(","):
            phrase = phrase.strip()
            if phrase:
                counter[phrase] += 1
    return counter.most_common(top_k)


def answer(raw_message, data_dir):
    text = raw_message.strip().lower()
    if not text:
        return HELP_TEXT

    if GREETING_PATTERN.match(text):
        return GREETING_TEXT
    if HELP_PATTERN.search(text):
        return HELP_TEXT

    wants_why = "why" in text

    id_match = _find_id(raw_message)
    if id_match:
        entity_key, prefix, num = id_match
        df, cfg = _load(entity_key, data_dir)
        # id zero-padding varies by entity (T0208 vs F172); compare as ints
        id_nums = df[cfg["id_col"]].astype(str).str.extract(r"(\d+)$")[0].astype(int)
        row = df[id_nums == num]
        if row.empty:
            return f"I don't see {prefix}{num:04d} in the {entity_key} data."
        row = row.iloc[0]
        flagged = "flagged for action" if row[cfg["flag_col"]] == 1 else "not flagged"
        reply = (
            f"{row[cfg['id_col']]}: {cfg['score_label']} {row[cfg['score_col']]:.1f}, "
            f"tier {row[cfg['tier_col']].capitalize()}, {flagged}."
        )
        if "top_reasons" in df.columns and pd.notna(row.get("top_reasons")):
            reply += f" Main reasons: {row['top_reasons']}."
        return reply

    entity_key = _find_entity(text)
    if entity_key is None:
        return HELP_TEXT

    df, cfg = _load(entity_key, data_dir)
    tier = _find_tier(text)
    wants_flagged = "flag" in text or "action" in text
    wants_average = "average" in text or "avg" in text or "mean" in text
    wants_list = any(w in text for w in ["which", "list", "show", "what are", "top"])

    subset = df
    label_bits = [entity_key + "s"]
    if tier:
        subset = subset[subset[cfg["tier_col"]] == tier]
        label_bits.append(f"in the {tier} tier")
    if wants_flagged:
        subset = subset[subset[cfg["flag_col"]] == 1]
        label_bits.append("flagged for action")
    label = " ".join(label_bits)

    if wants_why:
        if "top_reasons" not in df.columns:
            return f"I don't have reason data for {label}."
        if subset.empty:
            return f"No {label} to explain."
        reasons = _common_reasons(subset)
        if not reasons:
            return f"No consistent reasons stand out for {label}."
        formatted = ", ".join(f"{phrase} ({count})" for phrase, count in reasons)
        return f"Most common reasons for {label} ({len(subset)} total): {formatted}."

    if wants_average:
        if subset.empty:
            return f"No {label} to average."
        return f"Average {cfg['score_label']} for {label}: {subset[cfg['score_col']].mean():.1f}."

    if wants_list:
        if subset.empty:
            return f"No {label} right now."
        n = _find_top_n(text)
        top = subset.sort_values(cfg["score_col"], ascending=False).head(n)
        ids = ", ".join(top[cfg["id_col"]].astype(str))
        remaining = len(subset) - len(top)
        more = f" (+{remaining} more)" if remaining > 0 else ""
        return f"{len(subset)} {label}, top {len(top)} by {cfg['score_label']}: {ids}{more}"

    # default: a count
    return f"{len(subset)} {label} (out of {len(df)} total)."
