"""
SmartGrid PredictAI - Synthetic Data Generator
================================================
Builds larger, more realistic stand-ins for data/transformer_data.csv,
data/meter_data.csv, and data/feeder_data.csv, replacing the 5-row samples.
Column schemas match what notebooks/*.ipynb and models/*.py already expect -
only row counts and the realism of the underlying signal change.

Run from the repo root:
    python3 data/generate_data.py
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_TRANSFORMERS = 3000
N_METERS = 10000
N_FEEDERS = 500

# Asset-metadata/maintenance-history constants. REFERENCE_DATE is pinned
# rather than datetime.now() so regenerating this script in a later calendar
# year doesn't silently drift installation_year/service dates - matches the
# reproducibility goal of the fixed RANDOM_SEED.
REFERENCE_YEAR = 2026
REFERENCE_DATE = pd.Timestamp("2026-07-31")
# Tzaneen and the surrounding Mopani District villages/towns (Limpopo) -
# this prototype represents one regional utility office's service area, not
# all of South Africa. Kept at exactly 9 entries (the same length as the
# nine-province list this replaced) so rng.choice() draws the same number
# of indices from the stream either way - every column generated after
# `location` below (capacity_kva, previous_failures, service dates) stays
# byte-identical to before this change; only the location labels differ.
LOCATIONS = ["Tzaneen", "Nkowankowa", "Lenyenye", "Modjadjiskloof", "Haenertsburg",
             "Ga-Kgapane", "Bolobedu", "Xihoko", "Politsi"]
CAPACITY_KVA_OPTIONS = [50, 100, 200, 315, 500, 800, 1000, 1600, 2000]

# Grid topology: every feeder (and therefore every transformer on it) belongs
# to exactly one CNC (Control & Network Centre - the regional operations
# office) and one substation beneath that CNC. CNCs are the 9 LOCATIONS
# above; each gets a fixed number of substations.
SUBSTATIONS_PER_CNC = 3
CNC_CODES = {
    "Tzaneen": "TZN", "Nkowankowa": "NKW", "Lenyenye": "LNY",
    "Modjadjiskloof": "MDK", "Haenertsburg": "HBG", "Ga-Kgapane": "GKG",
    "Bolobedu": "BLB", "Xihoko": "XHK", "Politsi": "PLT",
}
# Approximate town-centre coordinates (Mopani District, Limpopo) used only as
# jitter anchors for synthetic substation/pole GPS - not surveyed asset
# locations.
CNC_COORDS = {
    "Tzaneen": (-23.8330, 30.1631),
    "Nkowankowa": (-23.9333, 30.2333),
    "Lenyenye": (-23.9833, 30.2667),
    "Modjadjiskloof": (-23.7333, 30.1667),
    "Haenertsburg": (-23.9667, 29.9667),
    "Ga-Kgapane": (-23.8667, 30.3333),
    "Bolobedu": (-23.9500, 30.4000),
    "Xihoko": (-23.7667, 30.3500),
    "Politsi": (-23.8167, 30.0333),
}


def generate_feeder_topology(n_feeders=N_FEEDERS, seed=RANDOM_SEED):
    """Assigns every feeder to one CNC and one substation beneath it. This is
    physical grid topology - it doesn't depend on any transformer/failure
    data, so it can be generated once and reused by both
    generate_transformer_data and generate_feeder_data as the single source
    of truth for "where is this asset" instead of each drawing its own
    independent, disconnected location."""
    rng = np.random.default_rng(seed + 3)
    feeder_ids = [f"F{i:03d}" for i in range(1, n_feeders + 1)]

    substations = []
    for cnc in LOCATIONS:
        code = CNC_CODES[cnc]
        base_lat, base_lon = CNC_COORDS[cnc]
        for s in range(1, SUBSTATIONS_PER_CNC + 1):
            substations.append({
                "cnc": cnc,
                "substation_id": f"SS-{code}-{s:02d}",
                "substation_name": f"{cnc} Substation {s}",
                "substation_lat": round(base_lat + rng.uniform(-0.05, 0.05), 5),
                "substation_lon": round(base_lon + rng.uniform(-0.05, 0.05), 5),
            })
    substations = pd.DataFrame(substations)

    # Round-robin then shuffle so every substation gets a near-even share of
    # feeders instead of a purely random (and therefore lumpy) draw.
    reps = int(np.ceil(n_feeders / len(substations)))
    assignment = np.tile(np.arange(len(substations)), reps)[:n_feeders]
    rng.shuffle(assignment)

    topo = substations.iloc[assignment].reset_index(drop=True)
    topo.insert(0, "feeder_id", feeder_ids)
    return topo


def generate_transformer_data(n=N_TRANSFORMERS, n_feeders=N_FEEDERS, seed=RANDOM_SEED, topology=None):
    rng = np.random.default_rng(seed)
    if topology is None:
        topology = generate_feeder_topology(n_feeders, seed)

    feeder_id = rng.integers(1, n_feeders + 1, size=n)
    age_years = np.clip(rng.gamma(shape=2.2, scale=6.0, size=n), 0, 35)
    load_factor = np.clip(rng.beta(2, 2, size=n) * 0.9 + 0.1, 0.1, 1.0)
    maintenance_score = np.clip(rng.beta(3, 2, size=n), 0, 1)
    oil_quality_index = np.clip(
        maintenance_score * 0.6 + rng.beta(3, 2, size=n) * 0.4, 0, 1
    )
    temperature_rise_c = np.clip(
        28
        + age_years * 0.6
        + load_factor * 22
        + (1 - oil_quality_index) * 18
        + rng.normal(0, 5, size=n),
        18,
        95,
    )

    # Logistic risk model: age, heavy load, poor maintenance/oil quality and
    # high temperature rise all push failure probability up.
    risk = (
        -8.5
        + age_years * 0.11
        + load_factor * 2.6
        + (1 - maintenance_score) * 4.2
        + (1 - oil_quality_index) * 3.4
        + (temperature_rise_c - 50) * 0.045
        + rng.normal(0, 0.35, size=n)
    )
    prob_failure = 1 / (1 + np.exp(-risk))
    failure_within_1yr = rng.binomial(1, prob_failure)

    # Asset metadata + maintenance history. Geography (cnc/substation) now
    # comes from the feeder topology map rather than an independent rng
    # draw, so a transformer's location always agrees with its feeder's -
    # this does shift the rng stream relative to older byte-identical runs,
    # but that guarantee only ever covered the pre-topology schema anyway.
    capacity_kva = rng.choice(CAPACITY_KVA_OPTIONS, size=n)
    installation_year = (REFERENCE_YEAR - age_years.round()).astype(int)

    previous_failures_lam = np.clip(
        0.15 + age_years / 35 * 1.5 + (1 - maintenance_score) * 1.0, 0.05, None
    )
    previous_failures = rng.poisson(previous_failures_lam)

    days_since_service = np.clip(
        rng.gamma(2.0, 60, size=n) * (1.6 - maintenance_score), 3, 730
    ).astype(int)
    last_serviced_date = REFERENCE_DATE - pd.to_timedelta(days_since_service, unit="D")

    days_since_oil = np.clip(
        rng.gamma(2.0, 150, size=n) * (1.6 - oil_quality_index), 14, 2000
    ).astype(int)
    last_oil_replacement_date = REFERENCE_DATE - pd.to_timedelta(days_since_oil, unit="D")

    feeder_id_str = [f"F{f:03d}" for f in feeder_id]
    topo_by_feeder = topology.set_index("feeder_id")
    topo_matched = topo_by_feeder.loc[feeder_id_str].reset_index()
    cnc = topo_matched["cnc"].values
    substation_id = topo_matched["substation_id"].values
    substation_name = topo_matched["substation_name"].values

    # Pole GPS: jitter a few hundred metres to ~1km around the transformer's
    # substation anchor. Drawn last so it doesn't disturb any column above.
    gps_lat = (topo_matched["substation_lat"].values + rng.normal(0, 0.006, size=n)).round(5)
    gps_lon = (topo_matched["substation_lon"].values + rng.normal(0, 0.006, size=n)).round(5)
    cnc_code = pd.Series(cnc).map(CNC_CODES).values
    pole_id = [f"POLE-{code}-{i:05d}" for code, i in zip(cnc_code, range(1, n + 1))]

    transformer_id = [f"T{i:04d}" for i in range(1, n + 1)]
    # Sequential-within-substation display name, e.g. "Tzaneen Substation 2
    # TX-014" - assigned after the substation is known, not part of the rng
    # stream (purely derived from substation_id order of appearance).
    seq_in_substation = (
        pd.Series(substation_id).groupby(pd.Series(substation_id)).cumcount() + 1
    ).values
    transformer_name = [
        f"{name} TX-{seq:03d}" for name, seq in zip(substation_name, seq_in_substation)
    ]

    df = pd.DataFrame({
        "transformer_id": transformer_id,
        "transformer_name": transformer_name,
        "feeder_id": feeder_id_str,
        "cnc": cnc,
        "substation_id": substation_id,
        "substation_name": substation_name,
        "age_years": age_years.round(1),
        "load_factor": load_factor.round(3),
        "maintenance_score": maintenance_score.round(3),
        "oil_quality_index": oil_quality_index.round(3),
        "temperature_rise_c": temperature_rise_c.round(1),
        "failure_within_1yr": failure_within_1yr,
        "pole_id": pole_id,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "capacity_kva": capacity_kva,
        "installation_year": installation_year,
        "previous_failures": previous_failures,
        "last_serviced_date": last_serviced_date.strftime("%Y-%m-%d"),
        "last_oil_replacement_date": last_oil_replacement_date.strftime("%Y-%m-%d"),
    })
    return df


def generate_meter_data(n=N_METERS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 1)

    area_theft_history_rate = np.clip(rng.beta(2, 10, size=n) * 0.6, 0.02, 0.35)
    theft_prob = np.clip(0.05 + area_theft_history_rate * 0.5, 0.03, 0.4)
    is_theft = rng.binomial(1, theft_prob)

    true_usage_kwh = np.clip(rng.gamma(shape=5, scale=25, size=n), 30, 400)

    # Honest meters: declared usage tracks true usage closely. Theft meters:
    # declared usage is under-reported relative to what the transformer
    # actually fed the connection - the classic bypass/tampering signature.
    underreport_factor = np.where(
        is_theft == 1,
        rng.uniform(1.3, 2.4, size=n),
        rng.uniform(0.97, 1.05, size=n),
    )
    declared_kwh = np.clip(true_usage_kwh / underreport_factor, 15, 400)
    transformer_feed_estimate_kwh = np.clip(
        true_usage_kwh + rng.normal(0, 6, size=n), 15, 450
    )
    historical_avg_kwh = np.clip(
        declared_kwh + rng.normal(0, 8, size=n), 10, 400
    )

    pct_drop_recent = np.clip(
        np.where(
            is_theft == 1,
            rng.uniform(0.15, 0.55, size=n),
            rng.uniform(0.0, 0.12, size=n),
        ),
        0,
        1,
    )
    night_usage_ratio = np.clip(
        np.where(
            is_theft == 1,
            rng.uniform(0.35, 0.7, size=n),
            rng.uniform(0.1, 0.4, size=n),
        ),
        0,
        1,
    )

    df = pd.DataFrame({
        "meter_id": [f"M{i:05d}" for i in range(1, n + 1)],
        "declared_kwh": declared_kwh.round(1),
        "transformer_feed_estimate_kwh": transformer_feed_estimate_kwh.round(1),
        "historical_avg_kwh": historical_avg_kwh.round(1),
        "pct_drop_recent": pct_drop_recent.round(3),
        "night_usage_ratio": night_usage_ratio.round(3),
        "area_theft_history_rate": area_theft_history_rate.round(3),
        "is_theft": is_theft,
    })
    return df


def generate_feeder_data(transformer_df, topology=None, n=N_FEEDERS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 2)
    if topology is None:
        topology = generate_feeder_topology(n, seed)

    feeder_id = [f"F{i:03d}" for i in range(1, n + 1)]

    # Ground-truth count of at-risk transformers per feeder, used only to
    # shape the outage label below - not exposed as a feature. The model
    # instead sees each feeder's *predicted* critical-tier transformer count
    # (models/outage_forecasting.py joins in transformer_risk_scores.csv),
    # which is a noisier, more realistic stand-in for this.
    risky_counts = (
        transformer_df[transformer_df["failure_within_1yr"] == 1]
        .groupby("feeder_id")
        .size()
        .reindex(feeder_id, fill_value=0)
        .values
    )

    feeder_age_years = np.clip(rng.gamma(shape=2.5, scale=7.0, size=n), 0, 40)
    vegetation_encroachment_score = np.clip(rng.beta(2, 5, size=n), 0, 1)
    protection_equipment_age = np.clip(rng.gamma(shape=2.0, scale=5.0, size=n), 0, 30)
    peak_load_pct = np.clip(rng.beta(2, 2, size=n) * 0.9 + 0.1, 0.1, 1.0)
    load_growth_rate = np.clip(rng.normal(0.04, 0.025, size=n), 0, 0.15)
    historical_outage_count_1yr = rng.poisson(lam=1.3, size=n)

    # Logistic risk model: aging/poorly-protected, overgrown, heavily-loaded
    # feeders with a history of outages and several at-risk transformers on
    # them are the ones likely to go out again soon.
    risk = (
        -9.3
        + feeder_age_years * 0.095
        + vegetation_encroachment_score * 4.4
        + protection_equipment_age * 0.075
        + peak_load_pct * 3.0
        + load_growth_rate * 13.0
        + historical_outage_count_1yr * 0.55
        + risky_counts * 0.30
        + rng.normal(0, 0.28, size=n)
    )
    prob_outage = 1 / (1 + np.exp(-risk))
    outage_within_7_days = rng.binomial(1, prob_outage)

    df = pd.DataFrame({
        "feeder_id": feeder_id,
        "feeder_age_years": feeder_age_years.round(1),
        "vegetation_encroachment_score": vegetation_encroachment_score.round(3),
        "protection_equipment_age": protection_equipment_age.round(1),
        "peak_load_pct": peak_load_pct.round(3),
        "load_growth_rate": load_growth_rate.round(3),
        "historical_outage_count_1yr": historical_outage_count_1yr,
        "outage_within_7_days": outage_within_7_days,
    })
    # Feeders inherit their CNC/substation from the same topology map
    # transformers use, so "CNC: Tzaneen, Feeder: F007" always refers to a
    # feeder that's actually physically under Tzaneen.
    return topology.merge(df, on="feeder_id")


def generate_transformer_history(transformer_df, n_months=12, seed=45):
    """12 months of oil_quality_index/temperature_rise_c/load_factor per
    transformer, drifting from a randomized starting point toward each
    transformer's actual current value (+ noise) - a real generated time
    series, so "is this trending up or down" questions have real numbers
    behind them instead of invented wording."""
    rng = np.random.default_rng(seed)
    n = len(transformer_df)
    months = np.arange(n_months)
    frac = (months / (n_months - 1))[None, :]

    def series(end_vals, start_noise_sd, drift_noise_sd, lo, hi):
        end = end_vals.values[:, None]
        start = np.clip(end + rng.normal(0, start_noise_sd, size=(n, 1)), lo, hi)
        return np.clip(
            start + (end - start) * frac + rng.normal(0, drift_noise_sd, size=(n, n_months)),
            lo, hi,
        )

    oil = series(transformer_df["oil_quality_index"], 0.15, 0.02, 0, 1)
    temp = series(transformer_df["temperature_rise_c"], 8, 1.5, 18, 95)
    load = series(transformer_df["load_factor"], 0.12, 0.02, 0.1, 1.0)

    return pd.DataFrame({
        "transformer_id": np.repeat(transformer_df["transformer_id"].values, n_months),
        "month_offset": np.tile(months, n),
        "oil_quality_index": oil.round(3).ravel(),
        "temperature_rise_c": temp.round(1).ravel(),
        "load_factor": load.round(3).ravel(),
    })


if __name__ == "__main__":
    topology = generate_feeder_topology()
    transformer_df = generate_transformer_data(topology=topology)
    meter_df = generate_meter_data()
    feeder_df = generate_feeder_data(transformer_df, topology)
    history_df = generate_transformer_history(transformer_df)

    transformer_df.to_csv("data/transformer_data.csv", index=False)
    meter_df.to_csv("data/meter_data.csv", index=False)
    feeder_df.to_csv("data/feeder_data.csv", index=False)
    history_df.to_csv("data/transformer_history.csv", index=False)

    print(f"Saved data/transformer_data.csv -> {transformer_df.shape[0]} rows, "
          f"failure_within_1yr rate = {transformer_df['failure_within_1yr'].mean():.1%}")
    print(f"Saved data/meter_data.csv -> {meter_df.shape[0]} rows, "
          f"is_theft rate = {meter_df['is_theft'].mean():.1%}")
    print(f"Saved data/feeder_data.csv -> {feeder_df.shape[0]} rows, "
          f"outage_within_7_days rate = {feeder_df['outage_within_7_days'].mean():.1%}")
    print(f"Saved data/transformer_history.csv -> {history_df.shape[0]} rows "
          f"({transformer_df.shape[0]} transformers x 12 months)")
