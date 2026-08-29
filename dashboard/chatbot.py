"""
SmartGrid PredictAI - Dashboard Chat
=======================================
A small rule-based (not LLM-backed) query engine over the three score CSVs.
No external API, no network dependency, no cost - just keyword matching
against a question, mapped onto a pandas filter. Good enough for "how many
transformers are emergency?" or "why is T0208 high risk?" style questions;
not a general chatbot. Shared by both dashboard/app.py (Flask) and
dashboard/streamlit_app.py (Streamlit) - the same rules answer either UI.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

import knowledge_base
import report

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
        "score_col": "theft_risk_pct",
        "score_label": "theft risk %",
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

TIERS = ["low", "moderate", "elevated", "emergency"]
# Older/alternate wording someone might still say out loud even though the
# dashboard itself standardized on the four words above.
TIER_SYNONYMS = {
    "urgent": "emergency", "critical": "emergency", "high": "elevated", "medium": "moderate",
}

# Optional hyphen/space between the prefix letter and the digits, e.g.
# "T-102" as well as "T0208" - real ids in this project's data have no
# separator, but that's not how everyone types them.
ID_PATTERN = re.compile(r"\b([TMF])[-\s]?0*(\d{1,5})\b", re.IGNORECASE)
TOP_N_PATTERN = re.compile(r"\btop\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:highest|riskiest|worst|most)\b")
GREETING_PATTERN = re.compile(r"^\s*(hi|hello|hey|howdy|yo)\b[!.\s]*$")
HELP_PATTERN = re.compile(r"\b(help|what can you (do|ask)|options|commands)\b")
# Follow-up references to the specific id from a prior turn, e.g. "why is
# it flagged?" or "what about that one?" right after asking about T0208.
PRONOUN_PATTERN = re.compile(r"\b(it|that one|this one)\b")
# "high-risk"/"high risk" as a two-tier alias (elevated + emergency), checked
# ahead of the single-tier TIER_SYNONYMS "high" -> "elevated" mapping.
HIGH_RISK_PATTERN = re.compile(r"\bhigh[\s-]?risk\b")

# Transformer-only id-scoped intents, matched as whole words so common words
# like "research" don't false-trigger "search". Order is the priority when
# more than one keyword appears in the same message.
TRANSFORMER_ONLY_KEYWORDS = (
    "predict", "health", "recommend", "history", "find", "search", "where", "trend", "report",
)

# 3-value Healthy/Warning/Emergency status now comes from the "status" column
# models/failure_prediction.py and models/outage_forecasting.py write into
# the scores CSVs (derived from the 4-tier risk_tier - see
# models/failure_prediction.py's TIER_TO_STATUS).

HELP_TEXT = (
    "I can answer questions about transformers, meters, and feeders. Try things like:\n"
    "• “how many transformers are emergency?”\n"
    "• “what's the risk score for T0208?”\n"
    "• “why is T0208 high risk?”\n"
    "• “why are feeders emergency?” (most common reasons across a group)\n"
    "• “top 5 riskiest meters”\n"
    "• “average theft risk for meters”\n"
    "• “which feeders are flagged?”\n"
    "• “which transformers are high-risk?” (elevated or emergency)\n"
    "Transformer-specific: “predict T0208”, “health of T0208”, “recommend "
    "actions for T0208”, “maintenance history of T0208”, “find T0208”, "
    "“trend for T0208”, “compare T0208 and T0301”, “generate report for T0208”\n"
    "Engineering reference (no id needed): “what is dissolved gas analysis?”, "
    "“transformer is making a loud humming noise”\n"
    "You can also ask follow-ups without repeating yourself, e.g. "
    "“why is T0208 high risk?” then “is it flagged?”"
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


def _find_all_ids(raw_text):
    """Every id mentioned in the message, order-preserving and deduped -
    used by the compare intent, which needs two ids from one message."""
    found = []
    for m in ID_PATTERN.finditer(raw_text):
        prefix, digits = m.group(1).upper(), int(m.group(2))
        for key, cfg in ENTITIES.items():
            if cfg["id_prefix"] == prefix:
                item = (key, prefix, digits)
                if item not in found:
                    found.append(item)
                break
    return found


def resolve_id(entity_key, num, data_dir):
    """Public counterpart to the id lookup answer() does inline - given an
    entity key and the numeric part of an id (e.g. ("transformer", 208)),
    returns the real, zero-padded id string from the CSV (e.g. "T0208"), or
    None if that number isn't in the data. Lets a caller that already has a
    context dict (e.g. the Flask session) resolve context["last_id"] into a
    display id without re-running answer()."""
    df, cfg = _load(entity_key, data_dir)
    id_nums = df[cfg["id_col"]].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    match = df[id_nums == num]
    return None if match.empty else match.iloc[0][cfg["id_col"]]


def _has_word(text, word):
    return re.search(rf"\b{word}\b", text) is not None


def _load_transformer_full(data_dir):
    """Merges the model's score output with the raw asset/maintenance
    columns generate_data.py produces, so id-scoped transformer intents
    (predict/health/recommend/history/find/trend/compare) can draw on both
    without either file duplicating the other's columns."""
    scores = pd.read_csv(data_dir / "transformer_risk_scores.csv")
    raw_cols = [
        "transformer_id", "transformer_name", "cnc", "substation_id",
        "substation_name", "pole_id", "gps_lat", "gps_lon",
        "age_years", "load_factor", "maintenance_score",
        "oil_quality_index", "temperature_rise_c", "capacity_kva",
        "installation_year", "previous_failures", "last_serviced_date",
        "last_oil_replacement_date",
    ]
    raw = pd.read_csv(data_dir / "transformer_data.csv")[raw_cols]
    return scores.merge(raw, on="transformer_id")


def _load_feeder_full(data_dir):
    """Same idea as _load_transformer_full: merges feeder_data.csv (raw
    topology - cnc/substation - plus grid characteristics) with
    feeder_outage_scores.csv (model output) on feeder_id."""
    scores = pd.read_csv(data_dir / "feeder_outage_scores.csv")
    raw = pd.read_csv(data_dir / "feeder_data.csv")
    return raw.merge(scores, on="feeder_id")


def _find_transformer_row(full, num):
    id_nums = full["transformer_id"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    match = full[id_nums == num]
    return None if match.empty else match.iloc[0]


def _trend_direction(values, threshold):
    slope = np.polyfit(np.arange(len(values)), values, 1)[0]
    if abs(slope) < threshold:
        return "stable"
    return "rising" if slope > 0 else "falling"


def _risk_trend_summary(transformer_id, data_dir):
    hist = pd.read_csv(data_dir / "transformer_history.csv")
    rows = hist[hist["transformer_id"] == transformer_id].sort_values("month_offset")
    if rows.empty:
        return "no trend data available"
    oil_dir = _trend_direction(rows["oil_quality_index"].values, 0.01)
    temp_dir = _trend_direction(rows["temperature_rise_c"].values, 0.5)
    load_dir = _trend_direction(rows["load_factor"].values, 0.01)
    return f"oil quality {oil_dir}, temperature {temp_dir}, load {load_dir}"


def _predict_reply(transformer_id, num, data_dir):
    full = _load_transformer_full(data_dir)
    row = _find_transformer_row(full, num)
    if row is None:
        return f"I don't have score data for {transformer_id} yet - run models/failure_prediction.py first."
    tier = row["risk_tier"]
    actions = knowledge_base.MAINTENANCE_ACTIONS.get(tier)
    recommendation = actions[0] if actions else "No recommendation available."
    return (
        f"{row['transformer_id']} prediction:\n"
        f"Failure Risk: {row['risk_score']:.1f}%\n"
        f"Risk Level: {tier.capitalize()}\n"
        f"Predicted Failure: {row['predicted_failure_mode']}\n"
        f"Confidence: {row['confidence_pct']:.1f}%\n"
        f"Recommendation: {recommendation}"
    )


def _health_reply(transformer_id, num, data_dir):
    full = _load_transformer_full(data_dir)
    row = _find_transformer_row(full, num)
    if row is None:
        return f"I don't have score data for {transformer_id} yet - run models/failure_prediction.py first."
    tier = row["risk_tier"]
    return (
        f"{row['transformer_id']} health:\n"
        f"Health Score: {row['health_score']:.1f}/100\n"
        f"Status: {row['status']}\n"
        f"Last Serviced: {row['last_serviced_date']}\n"
        f"Last Oil Replacement: {row['last_oil_replacement_date']}\n"
        f"Next Maintenance: {row['next_maintenance_date']}\n"
        f"Remaining Useful Life: {row['remaining_useful_life_years']:.1f} years\n"
        f"Risk Trend: {_risk_trend_summary(row['transformer_id'], data_dir)}"
    )


def _recommend_reply(transformer_id, num, data_dir):
    full = _load_transformer_full(data_dir)
    row = _find_transformer_row(full, num)
    if row is None:
        return f"I don't have score data for {transformer_id} yet - run models/failure_prediction.py first."
    tier = row["risk_tier"]
    actions = knowledge_base.MAINTENANCE_ACTIONS.get(tier, [])
    if not actions:
        return f"No recommended actions found for {row['transformer_id']}."
    bullets = "\n".join(f"- {a}" for a in actions)
    return f"Recommended actions for {row['transformer_id']} ({tier} tier):\n{bullets}"


def _history_reply(transformer_id, num, data_dir):
    full = _load_transformer_full(data_dir)
    row = _find_transformer_row(full, num)
    if row is None:
        return f"I don't have data for {transformer_id}."
    hist = pd.read_csv(data_dir / "transformer_history.csv")
    recent = hist[hist["transformer_id"] == row["transformer_id"]].sort_values("month_offset").tail(3)
    readings = "; ".join(
        f"month {int(r.month_offset)}: oil={r.oil_quality_index:.2f}, "
        f"temp={r.temperature_rise_c:.1f}°C, load={r.load_factor:.2f}"
        for r in recent.itertuples()
    )
    return (
        f"{row['transformer_id']} maintenance history:\n"
        f"Last Serviced: {row['last_serviced_date']}\n"
        f"Last Oil Replacement: {row['last_oil_replacement_date']}\n"
        f"Previous Failures: {int(row['previous_failures'])}\n"
        f"Recent readings - {readings}"
    )


def _find_reply(transformer_id, num, data_dir):
    full = _load_transformer_full(data_dir)
    row = _find_transformer_row(full, num)
    if row is None:
        return f"I don't see {transformer_id} in the transformer data."
    flagged = "flagged for action" if row["alert_flag"] == 1 else "not flagged"
    return (
        f"{row['transformer_id']} ({row['transformer_name']}):\n"
        f"CNC: {row['cnc']}\n"
        f"Substation: {row['substation_name']}\n"
        f"Feeder: {row['feeder_id']}\n"
        f"Pole: {row['pole_id']} ({row['gps_lat']}, {row['gps_lon']})\n"
        f"Capacity: {row['capacity_kva']} kVA\n"
        f"Installed: {int(row['installation_year'])}\n"
        f"Status: {row['status']}\n"
        f"Current Tier: {row['risk_tier'].capitalize()}, {flagged}"
    )


def _trend_reply(transformer_id, num, data_dir):
    full = _load_transformer_full(data_dir)
    row = _find_transformer_row(full, num)
    if row is None:
        return f"I don't have data for {transformer_id}."
    return (
        f"{row['transformer_id']} trend (last 12 months): "
        f"{_risk_trend_summary(row['transformer_id'], data_dir)}. "
        f"Current tier: {row['risk_tier'].capitalize()} (risk score {row['risk_score']:.1f})."
    )


def _report_reply(transformer_id, num, data_dir):
    try:
        path = report.generate_pdf_report(transformer_id, data_dir)
    except ValueError:
        return f"I don't have score data for {transformer_id} yet - run models/failure_prediction.py first."
    return (
        f"Saved a maintenance report for {transformer_id} to {path}. "
        f"In the Flask dashboard, open it at /reports/{path.name}."
    )


def _compare_reply(nums, data_dir):
    full = _load_transformer_full(data_dir)
    rows = []
    for num in nums:
        row = _find_transformer_row(full, num)
        if row is None:
            return f"I don't see a transformer numbered {num} in the data."
        rows.append(row)
    a, b = rows[0], rows[1]
    return "\n".join([
        f"Comparing {a['transformer_id']} vs {b['transformer_id']}:",
        f"Risk score: {a['risk_score']:.1f} vs {b['risk_score']:.1f}",
        f"Tier: {a['risk_tier'].capitalize()} vs {b['risk_tier'].capitalize()}",
        f"Health score: {a['health_score']:.1f} vs {b['health_score']:.1f}",
        f"Temperature: {a['temperature_rise_c']:.1f}°C vs {b['temperature_rise_c']:.1f}°C",
        f"Load factor: {a['load_factor']:.2f} vs {b['load_factor']:.2f}",
        f"Predicted failure: {a['predicted_failure_mode']} vs {b['predicted_failure_mode']}",
    ])


def _glossary_reply(term):
    entry = knowledge_base.GLOSSARY[term]
    causes = "; ".join(entry["causes"]) if entry["causes"] else "n/a"
    actions = "; ".join(entry["recommended_actions"])
    return (
        f"{term}:\n"
        f"Meaning: {entry['meaning']}\n"
        f"Causes: {causes}\n"
        f"Consequences: {entry['consequences']}\n"
        f"Recommended actions: {actions}"
    )


def _diagnosis_reply(symptom):
    entry = knowledge_base.SYMPTOM_CAUSES[symptom]
    causes = "\n".join(f"- {c}" for c in entry["causes"])
    steps = "\n".join(f"- {s}" for s in entry["inspection_steps"])
    return (
        f"Possible causes for \"{symptom}\":\n{causes}\n"
        f"Recommended inspection steps:\n{steps}"
    )


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


def answer(raw_message, data_dir, context=None):
    """context, if given, is a plain dict the caller persists across turns
    (Flask session, Streamlit session_state, ...) so follow-up questions can
    reuse the last id/entity mentioned. Mutated in place; omit it (or pass
    None) for a one-shot, stateless call."""
    if context is None:
        context = {}

    text = raw_message.strip().lower()
    if not text:
        return HELP_TEXT

    if GREETING_PATTERN.match(text):
        return GREETING_TEXT
    if HELP_PATTERN.search(text):
        return HELP_TEXT

    wants_why = "why" in text

    if _has_word(text, "compare"):
        compare_nums = [num for (key, _, num) in _find_all_ids(raw_message) if key == "transformer"]
        if len(compare_nums) >= 2:
            return _compare_reply(compare_nums[:2], data_dir)

    # Symptom/glossary lookups are checked ahead of id/entity dispatch so a
    # sentence like "transformer is making a loud humming noise" is caught
    # here rather than being swallowed by the generic entity-question path
    # below just because it contains the word "transformer".
    for term in sorted(knowledge_base.GLOSSARY, key=len, reverse=True):
        if term in text:
            return _glossary_reply(term)
    for symptom in sorted(knowledge_base.SYMPTOM_CAUSES, key=len, reverse=True):
        if symptom in text:
            return _diagnosis_reply(symptom)

    id_match = _find_id(raw_message)
    if id_match is None and PRONOUN_PATTERN.search(text) and context.get("last_id"):
        id_match = context["last_id"]
    if id_match:
        entity_key, prefix, num = id_match
        df, cfg = _load(entity_key, data_dir)
        # id zero-padding varies by entity (T0208 vs F172); compare as ints
        id_nums = df[cfg["id_col"]].astype(str).str.extract(r"(\d+)$")[0].astype(int)
        row = df[id_nums == num]
        if row.empty:
            return f"I don't see {prefix}{num:04d} in the {entity_key} data."
        context["last_id"] = [entity_key, prefix, num]
        context["last_entity"] = entity_key

        display_id = row.iloc[0][cfg["id_col"]]
        matched_keyword = next((k for k in TRANSFORMER_ONLY_KEYWORDS if _has_word(text, k)), None)
        if matched_keyword and entity_key != "transformer":
            return f"{display_id}: that's not available for meters/feeders yet - only transformers."
        if matched_keyword:
            if matched_keyword == "predict":
                return _predict_reply(display_id, num, data_dir)
            if matched_keyword == "health":
                return _health_reply(display_id, num, data_dir)
            if matched_keyword == "recommend":
                return _recommend_reply(display_id, num, data_dir)
            if matched_keyword == "history":
                return _history_reply(display_id, num, data_dir)
            if matched_keyword in ("find", "search", "where"):
                return _find_reply(display_id, num, data_dir)
            if matched_keyword == "trend":
                return _trend_reply(display_id, num, data_dir)
            if matched_keyword == "report":
                return _report_reply(display_id, num, data_dir)

        row = row.iloc[0]
        flagged = "flagged for action" if row[cfg["flag_col"]] == 1 else "not flagged"
        reply = (
            f"{row[cfg['id_col']]}: {cfg['score_label']} {row[cfg['score_col']]:.1f}, "
            f"tier {row[cfg['tier_col']].capitalize()}, {flagged}."
        )
        if "top_reasons" in df.columns and pd.notna(row.get("top_reasons")):
            reply += f" Main reasons: {row['top_reasons']}."
        return reply

    entity_key = _find_entity(text) or context.get("last_entity")
    if entity_key is None:
        return HELP_TEXT
    context["last_entity"] = entity_key

    df, cfg = _load(entity_key, data_dir)
    high_risk = HIGH_RISK_PATTERN.search(text) is not None
    tier = None if high_risk else _find_tier(text)
    wants_flagged = "flag" in text or "action" in text
    wants_average = "average" in text or "avg" in text or "mean" in text
    wants_list = any(w in text for w in ["which", "list", "show", "what are", "top"])

    subset = df
    label_bits = [entity_key + "s"]
    if high_risk:
        subset = subset[subset[cfg["tier_col"]].isin(["elevated", "emergency"])]
        label_bits.append("at high risk (elevated or emergency)")
    elif tier:
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
