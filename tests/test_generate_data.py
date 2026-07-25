import generate_data as gd

TRANSFORMER_COLUMNS = {
    "transformer_id", "age_years", "load_factor", "maintenance_score",
    "oil_quality_index", "temperature_rise_c", "failure_within_1yr",
}
METER_COLUMNS = {
    "meter_id", "declared_kwh", "transformer_feed_estimate_kwh",
    "historical_avg_kwh", "pct_drop_recent", "night_usage_ratio",
    "area_theft_history_rate", "is_theft",
}


def test_transformer_data_schema_and_balance():
    df = gd.generate_transformer_data(n=200, seed=1)
    assert set(df.columns) == TRANSFORMER_COLUMNS
    assert df["transformer_id"].is_unique
    # Both classes must be present and neither dominant, or downstream
    # stratified train/test splits and evaluation metrics break.
    assert 0.05 < df["failure_within_1yr"].mean() < 0.5


def test_meter_data_schema_and_balance():
    df = gd.generate_meter_data(n=200, seed=1)
    assert set(df.columns) == METER_COLUMNS
    assert df["meter_id"].is_unique
    assert 0.05 < df["is_theft"].mean() < 0.5
