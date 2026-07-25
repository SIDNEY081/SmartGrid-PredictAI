from pathlib import Path

import pandas as pd
import chatbot

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def test_count_query_matches_real_data():
    df = pd.read_csv(DATA / "feeder_outage_scores.csv")
    expected = (df["risk_tier"] == "critical").sum()
    reply = chatbot.answer("how many feeders are critical?", DATA)
    assert str(expected) in reply
    assert "critical" in reply.lower()


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
