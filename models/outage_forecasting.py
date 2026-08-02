"""
Predict AI - Feeder Outage Forecasting Model
==============================================
Supervised classification model that predicts the probability a feeder
(the line serving a group of transformers/customers) will have an outage
within the next 7 days.

Unlike the other two models, this one consumes another model's output:
critical_transformer_count comes from data/transformer_risk_scores.csv
(models/failure_prediction.py), aggregated per feeder. Run
failure_prediction.py first.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, accuracy_score,
)
import joblib

import explain

FEATURES = [
    "feeder_age_years", "vegetation_encroachment_score", "protection_equipment_age",
    "peak_load_pct", "load_growth_rate", "historical_outage_count_1yr",
    "critical_transformer_count",
]
TARGET = "outage_within_7_days"

# PLACEHOLDER: flag the riskiest 15% of feeders for pre-emptive attention
# (crew dispatch, vegetation clearing, load shedding review). Stand-in for
# real capacity - replace once known. Matches models/failure_prediction.py.
CAPACITY_FRACTION = 0.15

# Numeric companion to risk_tier so BI tools (Power BI, Tableau) can sort by
# severity instead of alphabetically (Critical, Elevated, Low, Moderate).
TIER_ORDER = {"low": 1, "moderate": 2, "elevated": 3, "critical": 4}


def load_data(
    feeder_path="data/feeder_data.csv",
    transformer_scores_path="data/transformer_risk_scores.csv",
):
    feeder_df = pd.read_csv(feeder_path)
    risk_df = pd.read_csv(transformer_scores_path)

    critical_counts = (
        risk_df[risk_df["risk_tier"] == "critical"]
        .groupby("feeder_id")
        .size()
        .rename("critical_transformer_count")
    )
    feeder_df = feeder_df.merge(critical_counts, on="feeder_id", how="left")
    feeder_df["critical_transformer_count"] = feeder_df["critical_transformer_count"].fillna(0)
    return feeder_df


def build_model():
    # Same capacity as failure_prediction.py - with 500 feeders to train on
    # there's enough data that shallower/fewer trees (tuned for an earlier
    # 200-feeder dataset) were underfitting instead of helping.
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=42)


def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    eval_model = build_model()
    eval_model.fit(X_train, y_train)

    y_proba = eval_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"Held-out ROC-AUC: {auc:.3f}, Average Precision: {ap:.3f}")

    y_pred = (y_proba >= 0.5).astype(int)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=["no_outage", "outage_risk"]))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix [[TN FP] [FN TP]] (rows=actual, cols=predicted):\n{cm}")

    # Refit on the full dataset for deployment scoring.
    model = build_model()
    model.fit(X, y)

    importance = explain.global_importance(model, FEATURES)
    print("Global feature importance:")
    print(importance.to_string(index=False))

    return model


def score_all_feeders(model, df, capacity_fraction=CAPACITY_FRACTION, explain_predictions=True):
    proba = model.predict_proba(df[FEATURES])[:, 1]
    result = df[["feeder_id"]].copy()
    result["outage_risk_score"] = (proba * 100).round(1)
    result["risk_tier"] = pd.cut(
        result["outage_risk_score"], bins=[-0.1, 20, 50, 75, 100],
        labels=["low", "moderate", "elevated", "critical"]
    )
    result["risk_tier_order"] = result["risk_tier"].map(TIER_ORDER)

    cutoff = result["outage_risk_score"].quantile(1 - capacity_fraction)
    result["alert_flag"] = (result["outage_risk_score"] >= cutoff).astype(int)

    if explain_predictions:
        contributions, _ = explain.explain_batch(model, df[FEATURES], FEATURES)
        result["top_reasons"] = [
            explain.describe_reasons(explain.top_reasons(contributions.loc[i]))
            for i in df.index
        ]

    return result.sort_values("outage_risk_score", ascending=False)


if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    scored = score_all_feeders(model, df)
    scored.to_csv("data/feeder_outage_scores.csv", index=False)
    joblib.dump(model, "models/outage_model.joblib")
    print("Saved model -> models/outage_model.joblib")
    print("Saved scores -> data/feeder_outage_scores.csv")
