import generate_data as gd

TRANSFORMER_COLUMNS = {
    "transformer_id", "transformer_name", "feeder_id", "cnc", "substation_id",
    "substation_name", "age_years", "load_factor", "maintenance_score",
    "oil_quality_index", "temperature_rise_c", "failure_within_1yr",
    "pole_id", "gps_lat", "gps_lon",
    "capacity_kva", "installation_year", "previous_failures",
    "last_serviced_date", "last_oil_replacement_date",
}
TRANSFORMER_HISTORY_COLUMNS = {
    "transformer_id", "month_offset", "oil_quality_index",
    "temperature_rise_c", "load_factor",
}
METER_COLUMNS = {
    "meter_id", "declared_kwh", "transformer_feed_estimate_kwh",
    "historical_avg_kwh", "pct_drop_recent", "night_usage_ratio",
    "area_theft_history_rate", "is_theft",
}
FEEDER_COLUMNS = {
    "feeder_id", "cnc", "substation_id", "substation_name",
    "substation_lat", "substation_lon",
    "feeder_age_years", "vegetation_encroachment_score",
    "protection_equipment_age", "peak_load_pct", "load_growth_rate",
    "historical_outage_count_1yr", "outage_within_7_days",
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


def test_feeder_data_schema_and_balance():
    topology = gd.generate_feeder_topology(n_feeders=100, seed=1)
    transformer_df = gd.generate_transformer_data(n=400, n_feeders=100, seed=1, topology=topology)
    df = gd.generate_feeder_data(transformer_df, topology, n=100, seed=1)
    assert set(df.columns) == FEEDER_COLUMNS
    assert df["feeder_id"].is_unique
    assert 0.05 < df["outage_within_7_days"].mean() < 0.5


def test_feeder_topology_every_feeder_assigned():
    topology = gd.generate_feeder_topology(n_feeders=100, seed=1)
    assert topology["feeder_id"].is_unique
    assert len(topology) == 100
    assert set(topology["cnc"]) <= set(gd.LOCATIONS)


def test_transformer_installation_year_matches_age():
    df = gd.generate_transformer_data(n=200, seed=1)
    # within 1 year, not exact: installation_year is derived from the raw,
    # unrounded age_years, while this recomputes from the display-rounded
    # (1dp) column, so values sitting exactly on a .5 tie can round
    # differently (double-rounding) without indicating an actual bug.
    implied_age = gd.REFERENCE_YEAR - df["installation_year"]
    assert (implied_age - df["age_years"].round()).abs().le(1).all()


def test_transformer_history_schema_and_shape():
    transformer_df = gd.generate_transformer_data(n=50, seed=1)
    history_df = gd.generate_transformer_history(transformer_df, n_months=12, seed=1)
    assert set(history_df.columns) == TRANSFORMER_HISTORY_COLUMNS
    assert len(history_df) == 50 * 12
    assert set(history_df["month_offset"]) == set(range(12))
    assert history_df.groupby("transformer_id").size().eq(12).all()
    assert history_df["oil_quality_index"].between(0, 1).all()
    assert history_df["temperature_rise_c"].between(18, 95).all()
    assert history_df["load_factor"].between(0.1, 1.0).all()
