"""
SmartGrid PredictAI - Synthetic Data Generator
================================================
Builds larger, more realistic stand-ins for data/transformer_data.csv and
data/meter_data.csv, replacing the 5-row samples. Column schemas match what
notebooks/*.ipynb and models/*.py already expect - only row counts and the
realism of the underlying signal change.

Run from the repo root:
    python3 data/generate_data.py
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_TRANSFORMERS = 800
N_METERS = 3000


def generate_transformer_data(n=N_TRANSFORMERS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

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

    df = pd.DataFrame({
        "transformer_id": [f"T{i:04d}" for i in range(1, n + 1)],
        "age_years": age_years.round(1),
        "load_factor": load_factor.round(3),
        "maintenance_score": maintenance_score.round(3),
        "oil_quality_index": oil_quality_index.round(3),
        "temperature_rise_c": temperature_rise_c.round(1),
        "failure_within_1yr": failure_within_1yr,
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


if __name__ == "__main__":
    transformer_df = generate_transformer_data()
    meter_df = generate_meter_data()

    transformer_df.to_csv("data/transformer_data.csv", index=False)
    meter_df.to_csv("data/meter_data.csv", index=False)

    print(f"Saved data/transformer_data.csv -> {transformer_df.shape[0]} rows, "
          f"failure_within_1yr rate = {transformer_df['failure_within_1yr'].mean():.1%}")
    print(f"Saved data/meter_data.csv -> {meter_df.shape[0]} rows, "
          f"is_theft rate = {meter_df['is_theft'].mean():.1%}")
