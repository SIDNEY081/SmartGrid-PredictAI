"""
Predict AI - Theft & Anomaly Detection Model
=============================================
Unsupervised anomaly detection on smart meter consumption data to flag
potential illegal connections, meter tampering, or theft.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
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
    "meter_reversal_events_6mo",
    "zero_consumption_days_90d",
    "tamper_alarm_count",
]
TARGET = "is_theft"  # used to train the supervised classifier and to evaluate both models

# Numeric companion to priority_tier so BI tools (Power BI, Tableau) can sort
# by severity instead of alphabetically (Emergency, Elevated, Low, Moderate).
TIER_ORDER = {"low": 1, "moderate": 2, "elevated": 3, "emergency": 4}

# Weight given to the supervised classifier's probability vs. the unsupervised
# Isolation Forest's anomaly score when blending into theft_risk_pct. Weighted
# toward supervised since it's calibrated against real labels; the anomaly
# score is kept in the mix so a pattern the classifier wasn't trained on can
# still push the blended score up.
SUPERVISED_BLEND_WEIGHT = 0.6

# PLACEHOLDER: a representative South African residential tariff, not a real
# Eskom/municipal rate for this service area - swap for the real figure once
# known. Used only to turn diverted kWh into a rand estimate for the
# Executive Overview; label it as a prototype estimate wherever it's shown.
TARIFF_RAND_PER_KWH = 2.60

THEFT_ACTIONS = {
    "emergency": [
        "Dispatch field inspection within 24 hours",
        "Cross-check transformer feed log against billed usage",
        "Escalate to Revenue Protection for audit",
    ],
    "elevated": [
        "Schedule field inspection within 7 days",
        "Review recent meter reading history for irregularities",
        "Notify the area Revenue Protection Officer",
    ],
    "moderate": [
        "Monitor consumption trend at the next scheduled read",
        "Flag account for a routine audit",
    ],
    "low": [
        "No action required",
        "Continue routine billing checks",
    ],
}


def load_data(path="data/meter_data.csv"):
    df = pd.read_csv(path)
    df["feed_vs_declared_ratio"] = df["transformer_feed_estimate_kwh"] / df["declared_kwh"].clip(lower=1)
    df["expected_kwh"] = df["transformer_feed_estimate_kwh"]
    df["actual_kwh"] = df["declared_kwh"]
    df["consumption_deviation_pct"] = (
        (df["expected_kwh"] - df["actual_kwh"]) / df["expected_kwh"].clip(lower=1) * 100
    )
    return df


def train_model(df, contamination=0.08):
    feature_cols = NUMERIC_FEATURES + ["feed_vs_declared_ratio"]
    X = df[feature_cols]
    y = df[TARGET]

    # --- Unsupervised signal: Isolation Forest anomaly score. Kept even now
    # that a supervised model exists below, because it can still flag a
    # genuinely novel consumption pattern the classifier was never trained
    # to recognize. ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso_model.fit(X_scaled)

    raw_scores = -iso_model.score_samples(X_scaled)
    anomaly_score_0_100 = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min()) * 100
    flagged = iso_model.predict(X_scaled) == -1

    auc = roc_auc_score(y, anomaly_score_0_100)
    ap = average_precision_score(y, anomaly_score_0_100)
    print(f"[Isolation Forest, unsupervised] ROC-AUC: {auc:.3f}, Average Precision: {ap:.3f}")
    print(f"Accuracy: {accuracy_score(y, flagged.astype(int)):.3f}")
    print(classification_report(y, flagged.astype(int), target_names=["normal", "theft_flagged"]))
    cm = confusion_matrix(y, flagged.astype(int))
    print(f"Confusion matrix [[TN FP] [FN TP]] (rows=actual, cols=predicted):\n{cm}")

    # --- Supervised signal: Random Forest trained on the real is_theft
    # label, which the Isolation Forest above never sees during fitting.
    # Evaluated on a held-out split first (honest numbers for the pitch),
    # then refit on the full dataset for deployment scoring - same pattern
    # as models/failure_prediction.py's build_model/train_model split. ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    def build_classifier():
        return RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

    eval_classifier = build_classifier()
    eval_classifier.fit(X_train, y_train)
    y_proba_test = eval_classifier.predict_proba(X_test)[:, 1]
    y_pred_test = eval_classifier.predict(X_test)

    print(f"\n[Random Forest, supervised, held-out test split] "
          f"ROC-AUC: {roc_auc_score(y_test, y_proba_test):.3f}, "
          f"Precision: {precision_score(y_test, y_pred_test):.3f}, "
          f"Recall: {recall_score(y_test, y_pred_test):.3f}, "
          f"F1: {f1_score(y_test, y_pred_test):.3f}")
    print(classification_report(y_test, y_pred_test, target_names=["normal", "theft"]))
    print(f"Confusion matrix [[TN FP] [FN TP]] (rows=actual, cols=predicted):\n"
          f"{confusion_matrix(y_test, y_pred_test)}")

    classifier = build_classifier()
    classifier.fit(X, y)
    supervised_proba = classifier.predict_proba(X)[:, 1]

    importance = explain.global_importance(classifier, feature_cols)
    print("\nGlobal feature importance (Random Forest):")
    print(importance.to_string(index=False))

    theft_risk_pct = (
        SUPERVISED_BLEND_WEIGHT * supervised_proba * 100
        + (1 - SUPERVISED_BLEND_WEIGHT) * anomaly_score_0_100
    )

    return iso_model, classifier, scaler, feature_cols, anomaly_score_0_100, flagged, theft_risk_pct


def score_all_meters(
    df, scaler, feature_cols, anomaly_score_0_100, flagged, theft_risk_pct,
    model=None, classifier=None, explain_predictions=True,
):
    result = df[["meter_id"]].copy()
    result["anomaly_score"] = anomaly_score_0_100.round(1)
    result["investigation_flag"] = flagged.astype(int)
    result["theft_risk_pct"] = np.round(theft_risk_pct, 1)
    # Same four-level severity scale as the other two models (low/moderate/
    # elevated/emergency) - the labels used to differ ("urgent" instead of
    # "emergency") purely because this script was written independently of
    # failure_prediction.py, not because it means anything different.
    result["priority_tier"] = pd.cut(
        result["theft_risk_pct"], bins=[-0.1, 40, 65, 85, 100],
        labels=["low", "moderate", "elevated", "emergency"]
    )
    result["priority_tier_order"] = result["priority_tier"].map(TIER_ORDER)
    result["recommended_action"] = result["priority_tier"].astype(str).map(
        lambda tier: THEFT_ACTIONS.get(tier, [""])[0]
    )

    for col in (
        "expected_kwh", "actual_kwh", "consumption_deviation_pct",
        "feeder_id", "cnc", "substation_id", "substation_name",
        "confirmed_incidents_nearby_12mo",
    ):
        if col in df.columns:
            result[col] = df[col]

    if {"expected_kwh", "actual_kwh"} <= set(result.columns):
        result["estimated_monthly_loss_rand"] = (
            (result["expected_kwh"] - result["actual_kwh"]).clip(lower=0) * TARIFF_RAND_PER_KWH
        ).round(0)

    if explain_predictions and classifier is not None:
        contributions, _ = explain.explain_batch(classifier, df[feature_cols], feature_cols)
        result["top_reasons"] = [
            explain.describe_reasons(explain.top_reasons(contributions.loc[i]))
            for i in df.index
        ]

    return result.sort_values("theft_risk_pct", ascending=False)


if __name__ == "__main__":
    df = load_data()
    iso_model, classifier, scaler, feature_cols, anomaly_scores, flagged, theft_risk_pct = train_model(df)
    scored = score_all_meters(
        df, scaler, feature_cols, anomaly_scores, flagged, theft_risk_pct,
        model=iso_model, classifier=classifier,
    )
    scored.to_csv("data/meter_theft_scores.csv", index=False)
    joblib.dump(
        {"model": iso_model, "classifier": classifier, "scaler": scaler, "features": feature_cols},
        "models/theft_model.joblib",
    )
    print("Saved model -> models/theft_model.joblib")
    print("Saved scores -> data/meter_theft_scores.csv")
