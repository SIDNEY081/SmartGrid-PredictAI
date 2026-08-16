from pathlib import Path

import pandas as pd
import chatbot
import knowledge_base

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def test_count_query_matches_real_data():
    df = pd.read_csv(DATA / "feeder_outage_scores.csv")
    expected = (df["risk_tier"] == "emergency").sum()
    reply = chatbot.answer("how many feeders are emergency?", DATA)
    assert str(expected) in reply
    assert "emergency" in reply.lower()


def test_flagged_query_matches_real_data():
    df = pd.read_csv(DATA / "meter_theft_scores.csv")
    expected = int(df["investigation_flag"].sum())
    reply = chatbot.answer("which meters are flagged?", DATA)
    assert str(expected) in reply


def test_average_query_matches_real_data():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    expected = round(df["risk_score"].mean(), 1)
    reply = chatbot.answer("average risk score for transformers", DATA)
    assert f"{expected:.1f}" in reply


def test_id_lookup_matches_real_row():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"what's the risk score for {row['transformer_id']}?", DATA)
    assert row["transformer_id"] in reply
    assert f"{row['risk_score']:.1f}" in reply


def test_unknown_id_reports_not_found():
    reply = chatbot.answer("what about T9999?", DATA)
    assert "don't see" in reply.lower()


def test_unrecognized_question_returns_help():
    reply = chatbot.answer("blah unrelated nonsense", DATA)
    assert reply == chatbot.HELP_TEXT


def test_empty_message_returns_help():
    assert chatbot.answer("", DATA) == chatbot.HELP_TEXT


def test_why_id_query_surfaces_reasons():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"why is {row['transformer_id']} high risk?", DATA)
    assert "Main reasons" in reply
    assert row["top_reasons"] in reply


def test_why_aggregate_query_returns_common_reasons():
    df = pd.read_csv(DATA / "feeder_outage_scores.csv")
    emergency = df[df["risk_tier"] == "emergency"]
    reply = chatbot.answer("why are feeders emergency?", DATA)
    assert str(len(emergency)) in reply
    assert "Most common reasons" in reply


def test_top_n_query_respects_requested_count():
    reply = chatbot.answer("top 3 riskiest transformers", DATA)
    assert "top 3" in reply.lower()
    assert len(reply.split(":")[-1].split(",")) == 3


def test_greeting_returns_help():
    reply = chatbot.answer("hello", DATA)
    assert "transformers, meters, and feeders" in reply


def test_followup_pronoun_resolves_last_id():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    context = {}
    chatbot.answer(f"what's the risk score for {row['transformer_id']}?", DATA, context=context)
    reply = chatbot.answer("why is it flagged?", DATA, context=context)
    assert row["transformer_id"] in reply


def test_followup_without_context_has_no_pronoun_to_resolve():
    # No prior turn, so "it" has nothing to resolve to - falls back to help.
    reply = chatbot.answer("why is it flagged?", DATA)
    assert reply == chatbot.HELP_TEXT


def test_followup_omitted_entity_reuses_last_entity():
    context = {}
    chatbot.answer("how many transformers are emergency?", DATA, context=context)
    reply = chatbot.answer("which are flagged?", DATA, context=context)
    assert "transformer" in reply.lower()


def test_followup_context_isolated_per_conversation():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    context_a = {}
    context_b = {}
    chatbot.answer(f"what's the risk score for {row['transformer_id']}?", DATA, context=context_a)
    reply_b = chatbot.answer("why is it flagged?", DATA, context=context_b)
    assert reply_b == chatbot.HELP_TEXT


def test_hyphenated_id_is_recognized():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    num = int(row["transformer_id"][1:])
    reply = chatbot.answer(f"what's the risk score for T-{num}?", DATA)
    assert row["transformer_id"] in reply


def test_predict_reply_has_all_fields():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"predict {row['transformer_id']}", DATA)
    assert "Failure Risk" in reply
    assert "Risk Level" in reply
    assert "Predicted Failure" in reply
    assert "Confidence" in reply
    assert "Recommendation" in reply
    assert f"{row['risk_score']:.1f}" in reply


def test_health_reply_has_all_fields():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"health of {row['transformer_id']}", DATA)
    assert "Health Score" in reply
    assert "Status" in reply
    assert "Last Serviced" in reply
    assert "Remaining Useful Life" in reply
    assert "Risk Trend" in reply


def test_recommend_reply_lists_tier_actions():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"recommend actions for {row['transformer_id']}", DATA)
    assert "Recommended actions" in reply
    for action in knowledge_base.MAINTENANCE_ACTIONS[row["risk_tier"]]:
        assert action in reply


def test_history_reply_has_maintenance_facts():
    df = pd.read_csv(DATA / "transformer_data.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"maintenance history of {row['transformer_id']}", DATA)
    assert "Previous Failures" in reply
    assert str(row["previous_failures"]) in reply
    assert row["last_serviced_date"] in reply


def test_find_reply_has_asset_facts():
    df = pd.read_csv(DATA / "transformer_data.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"find {row['transformer_id']}", DATA)
    assert row["cnc"] in reply
    assert str(row["capacity_kva"]) in reply


def test_trend_reply_reports_a_direction():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"trend for {row['transformer_id']}", DATA)
    assert "trend" in reply.lower()
    assert any(word in reply for word in ["rising", "falling", "stable"])


def test_compare_reply_shows_both_ids():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    id_a, id_b = df.iloc[0]["transformer_id"], df.iloc[1]["transformer_id"]
    reply = chatbot.answer(f"compare {id_a} and {id_b}", DATA)
    assert id_a in reply
    assert id_b in reply
    assert "Risk score" in reply


def test_transformer_only_intent_guards_meters():
    df = pd.read_csv(DATA / "meter_theft_scores.csv")
    meter_id = df.iloc[0]["meter_id"]
    reply = chatbot.answer(f"predict {meter_id}", DATA)
    assert "not available" in reply.lower()


def test_glossary_lookup():
    reply = chatbot.answer("what is dissolved gas analysis?", DATA)
    assert "Meaning" in reply
    assert "Recommended actions" in reply


def test_symptom_diagnosis_lookup_even_when_message_says_transformer():
    # Regression: the message contains the word "transformer" (an entity
    # alias) but should still be caught as a symptom question, not routed
    # into the generic "how many transformers..." entity path.
    reply = chatbot.answer("transformer is making a loud humming noise", DATA)
    assert "Possible causes" in reply
    assert "inspection steps" in reply.lower()


def test_report_reply_mentions_pdf_path():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    row = df.iloc[0]
    reply = chatbot.answer(f"generate report for {row['transformer_id']}", DATA)
    assert row["transformer_id"] in reply
    assert ".pdf" in reply


def test_high_risk_alias_includes_elevated_and_emergency():
    df = pd.read_csv(DATA / "transformer_risk_scores.csv")
    expected = df["risk_tier"].isin(["elevated", "emergency"]).sum()
    reply = chatbot.answer("how many transformers are high-risk?", DATA)
    assert str(expected) in reply
