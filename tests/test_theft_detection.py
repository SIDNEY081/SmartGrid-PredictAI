from pathlib import Path

import theft_detection as td
import generate_data as gd

ROOT = Path(__file__).resolve().parent.parent


def test_load_data_has_required_columns():
    df = td.load_data()
    for col in td.NUMERIC_FEATURES + [td.TARGET]:
        assert col in df.columns, f"data/meter_data.csv is missing column {col!r} used by theft_detection.py"


def test_generator_schema_matches_model_features():
    # Guards against the schema drift that broke this script previously:
    # generate_data.py's output must keep providing every column
    # theft_detection.py expects.
    df = gd.generate_meter_data(n=50, seed=1)
    for col in td.NUMERIC_FEATURES + [td.TARGET]:
        assert col in df.columns


def test_pipeline_end_to_end():
    df = td.load_data()
    model, classifier, scaler, feature_cols, anomaly_scores, flagged, theft_risk_pct = td.train_model(df)
    assert len(anomaly_scores) == len(df)
    assert len(flagged) == len(df)
    assert len(theft_risk_pct) == len(df)

    scored = td.score_all_meters(
        df, scaler, feature_cols, anomaly_scores, flagged, theft_risk_pct, model=model, classifier=classifier
    )
    assert len(scored) == len(df)
    assert {
        "meter_id", "anomaly_score", "investigation_flag", "priority_tier", "top_reasons",
        "theft_risk_pct", "expected_kwh", "actual_kwh", "consumption_deviation_pct", "recommended_action",
        "estimated_monthly_loss_rand",
    } <= set(scored.columns)
    assert scored["meter_id"].is_unique


def test_output_path_does_not_collide_with_notebook_output():
    # theft_detection.py and notebooks/theft_detection_model.ipynb must write
    # to different files, or running one silently overwrites the other's
    # (differently-shaped) results.
    src = (ROOT / "models" / "theft_detection.py").read_text()
    assert "meter_theft_scores.csv" in src
    assert "meter_anomaly_scores.csv" not in src
