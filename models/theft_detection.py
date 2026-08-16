"""
Predict AI - Theft & Anomaly Detection Model
=============================================
Unsupervised anomaly detection on smart meter consumption data to flag
potential illegal connections, meter tampering, or theft.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, accuracy_score,
)
import joblib

import explain

NUMERIC_FEATURES = [
    "declared_kwh",
    "transformer_feed_estimate_kwh",
    "historical_avg_kwh",
    "pct_drop_recent",
    "night_usage_ratio",
    "area_theft_history_rate",
]
TARGET = "is_theft"  # used for evaluation only, not for training

# Numeric companion to priority_tier so BI tools (Power BI, Tableau) can sort
# by severity instead of alphabetically (Emergency, Elevated, Low, Moderate).
TIER_ORDER = {"low": 1, "moderate": 2, "elevated": 3, "emergency": 4}


def load_data(path="data/meter_data.csv"):
    df = pd.read_csv(path)
    df["feed_vs_declared_ratio"] = df["transformer_feed_estimate_kwh"] / df["declared_kwh"].clip(lower=1)
    return df


def train_model(df, contamination=0.08):
    feature_cols = NUMERIC_FEATURES + ["feed_vs_declared_ratio"]
    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    raw_scores = -model.score_samples(X_scaled)
    anomaly_score_0_100 = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min()) * 100

    auc = roc_auc_score(df[TARGET], anomaly_score_0_100)
    ap = average_precision_score(df[TARGET], anomaly_score_0_100)
    print(f"ROC-AUC: {auc:.3f}, Average Precision: {ap:.3f}")

    flagged = model.predict(X_scaled) == -1
    print(f"Accuracy: {accuracy_score(df[TARGET], flagged.astype(int)):.3f}")
    print(classification_report(df[TARGET], flagged.astype(int), target_names=["normal", "theft_flagged"]))
    cm = confusion_matrix(df[TARGET], flagged.astype(int))
    print(f"Confusion matrix [[TN FP] [FN TP]] (rows=actual, cols=predicted):\n{cm}")

    # Isolation Forest has no feature_importances_, so global importance is
    # derived from the same per-row contributions used for local explanations.
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    contributions, _ = explain.explain_batch(model, X_scaled_df, feature_cols)
    importance = explain.importance_from_contributions(contributions)
    print("Global feature importance:")
    print(importance.to_string(index=False))

    return model, scaler, feature_cols, anomaly_score_0_100, flagged


def score_all_meters(df, scaler, feature_cols, anomaly_score_0_100, flagged, model=None, explain_predictions=True):
    result = df[["meter_id"]].copy()
    result["anomaly_score"] = anomaly_score_0_100.round(1)
    result["investigation_flag"] = flagged.astype(int)
    # Same four-level severity scale as the other two models (low/moderate/
    # elevated/emergency) - the labels used to differ ("urgent" instead of
    # "emergency") purely because this script was written independently of
    # failure_prediction.py, not because it means anything different.
    result["priority_tier"] = pd.cut(
        result["anomaly_score"], bins=[-0.1, 40, 65, 85, 100],
        labels=["low", "moderate", "elevated", "emergency"]
    )
    result["priority_tier_order"] = result["priority_tier"].map(TIER_ORDER)

    if explain_predictions and model is not None:
        X_scaled = scaler.transform(df[feature_cols])
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
        contributions, _ = explain.explain_batch(model, X_scaled_df, feature_cols)
        result["top_reasons"] = [
            explain.describe_reasons(explain.top_reasons(contributions.loc[i]))
            for i in df.index
        ]

    return result.sort_values("anomaly_score", ascending=False)


if __name__ == "__main__":
    df = load_data()
    model, scaler, feature_cols, anomaly_scores, flagged = train_model(df)
    scored = score_all_meters(df, scaler, feature_cols, anomaly_scores, flagged, model=model)
    scored.to_csv("data/meter_theft_scores.csv", index=False)
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, "models/theft_model.joblib")
    print("Saved model -> models/theft_model.joblib")
    print("Saved scores -> data/meter_theft_scores.csv")
