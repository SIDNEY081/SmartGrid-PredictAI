"""
Predict AI - Transformer Failure Prediction Model
==================================================
Supervised classification model that predicts the probability a distribution
transformer will fail within the next 30 days.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, average_precision_score
import joblib

FEATURES = [
    "age_years", "load_factor", "maintenance_score",
    "oil_quality_index", "temperature_rise_c",
]
TARGET = "failure_within_1yr"

# PLACEHOLDER: flag the riskiest 15% of transformers for inspection. This is
# a stand-in for real maintenance-crew capacity (how many inspections can
# actually happen per week) - replace with a real number once known.
CAPACITY_FRACTION = 0.15

# Numeric companion to risk_tier so BI tools (Power BI, Tableau) can sort by
# severity instead of alphabetically (Critical, Elevated, Low, Moderate).
TIER_ORDER = {"low": 1, "moderate": 2, "elevated": 3, "critical": 4}


def load_data(path="data/transformer_data.csv"):
    return pd.read_csv(path)


def build_model(model_type):
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=42)
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)


def train_model(df, model_type="gradient_boosting"):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    eval_model = build_model(model_type)
    eval_model.fit(X_train, y_train)

    y_proba = eval_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"Held-out ROC-AUC: {auc:.3f}, Average Precision: {ap:.3f}")

    # Diagnostic only, not used for the deployment alert_flag below: shows
    # what precision costs to hit 85% recall, for comparing model runs.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    target_recall = 0.85
    idx = np.argmin(np.abs(recalls[:-1] - target_recall))
    chosen_threshold = thresholds[idx]
    print(f"[diagnostic] threshold for ~85% recall: {chosen_threshold:.3f}, Precision: {precisions[idx]:.3f}")

    y_pred = (y_proba >= chosen_threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=["healthy", "failure_risk"]))

    # Refit on the full dataset for deployment scoring.
    model = build_model(model_type)
    model.fit(X, y)

    return model, chosen_threshold


def score_all_transformers(model, df, capacity_fraction=CAPACITY_FRACTION):
    proba = model.predict_proba(df[FEATURES])[:, 1]
    id_cols = ["transformer_id", "feeder_id"] if "feeder_id" in df.columns else ["transformer_id"]
    result = df[id_cols].copy()
    result["risk_score"] = (proba * 100).round(1)
    result["risk_tier"] = pd.cut(result["risk_score"], bins=[-0.1, 20, 50, 75, 100], labels=["low", "moderate", "elevated", "critical"])
    result["risk_tier_order"] = result["risk_tier"].map(TIER_ORDER)

    # Capacity-based flag: the N riskiest transformers a crew could actually
    # inspect, rather than every transformer that clears a recall-tuned
    # probability threshold (that flagged ~60% of the fleet - not usable
    # as a work list).
    cutoff = result["risk_score"].quantile(1 - capacity_fraction)
    result["alert_flag"] = (result["risk_score"] >= cutoff).astype(int)

    return result.sort_values("risk_score", ascending=False)


if __name__ == "__main__":
    df = load_data()
    model, _ = train_model(df)
    scored = score_all_transformers(model, df)
    scored.to_csv("data/transformer_risk_scores.csv", index=False)
    joblib.dump(model, "models/failure_model.joblib")
    print("Saved model -> models/failure_model.joblib")
    print("Saved scores -> data/transformer_risk_scores.csv")
