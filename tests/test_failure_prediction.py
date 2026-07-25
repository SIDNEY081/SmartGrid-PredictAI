from pathlib import Path

import failure_prediction as fp
import generate_data as gd

ROOT = Path(__file__).resolve().parent.parent


def test_load_data_has_required_columns():
    df = fp.load_data()
    for col in fp.FEATURES + [fp.TARGET]:
        assert col in df.columns, f"data/transformer_data.csv is missing column {col!r} used by failure_prediction.py"


def test_generator_schema_matches_model_features():
    # Guards against the schema drift that broke this script previously:
    # generate_data.py's output must keep providing every column
    # failure_prediction.py expects.
    df = gd.generate_transformer_data(n=50, seed=1)
    for col in fp.FEATURES + [fp.TARGET]:
        assert col in df.columns


def test_pipeline_end_to_end():
    df = fp.load_data()
    model, threshold = fp.train_model(df)

    scored = fp.score_all_transformers(model, df, threshold)
    assert len(scored) == len(df)
    assert {"transformer_id", "risk_score", "risk_tier", "alert_flag"} <= set(scored.columns)
    assert scored["transformer_id"].is_unique


def test_output_path_does_not_collide_with_notebook_output():
    src = (ROOT / "models" / "failure_prediction.py").read_text()
    assert "transformer_risk_scores.csv" in src
    assert "transformer_failure_scores.csv" not in src
