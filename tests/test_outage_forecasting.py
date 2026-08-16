from pathlib import Path

import outage_forecasting as of
import generate_data as gd

ROOT = Path(__file__).resolve().parent.parent


def test_load_data_has_required_columns():
    df = of.load_data()
    for col in of.FEATURES + [of.TARGET]:
        assert col in df.columns, f"loaded feeder data is missing column {col!r} used by outage_forecasting.py"
    assert (df["emergency_transformer_count"] >= 0).all()


def test_generator_schema_matches_model_features():
    # Guards against schema drift: generate_data.py's feeder output must
    # keep providing every column outage_forecasting.py expects (aside from
    # emergency_transformer_count, which is joined in from
    # transformer_risk_scores.csv, not generated directly).
    transformer_df = gd.generate_transformer_data(n=100, n_feeders=25, seed=1)
    feeder_df = gd.generate_feeder_data(transformer_df, n=25, seed=1)
    expected = set(of.FEATURES) - {"emergency_transformer_count"}
    for col in expected | {of.TARGET}:
        assert col in feeder_df.columns


def test_pipeline_end_to_end():
    df = of.load_data()
    model = of.train_model(df)

    scored = of.score_all_feeders(model, df)
    assert len(scored) == len(df)
    assert {"feeder_id", "outage_risk_score", "risk_tier", "alert_flag"} <= set(scored.columns)
    assert scored["feeder_id"].is_unique


def test_alert_flag_matches_capacity_fraction():
    df = of.load_data()
    model = of.train_model(df)
    scored = of.score_all_feeders(model, df, capacity_fraction=0.15)
    flagged_rate = scored["alert_flag"].mean()
    assert abs(flagged_rate - 0.15) < 0.03
