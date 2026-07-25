"""
Predict AI - Transformer Failure Prediction Model
==================================================
Supervised classification model that predicts the probability a distribution
transformer will fail within the next 30 days.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, average_precision_score
import joblib

FEATURES = [
    "age_years", "load_factor", "maintenance_score",
    "oil_quality_index", "temperature_rise_c",
]
TARGET = "failure_within_1yr"


def load_data(path="data/transformer_data.csv"):
    return pd.read_csv(path)


def train_model(df, model_type="gradient_boosting"):
    # NOTE: the sample dataset in data/transformer_data.csv is a handful of
    # rows, too small for a held-out test split to contain both classes.
    # Train and evaluate in-sample here (matching notebooks/transformer_failure_model.ipynb);
    # switch to a train/test split once a production-sized dataset is available.
    X, y = df[FEATURES], df[TARGET]

    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)

    model.fit(X, y)

    y_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_proba)
    ap = average_precision_score(y, y_proba)
    print(f"ROC-AUC: {auc:.3f}, Average Precision: {ap:.3f}")

    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    target_recall = 0.85
    idx = np.argmin(np.abs(recalls[:-1] - target_recall))
    chosen_threshold = thresholds[idx]
    print(f"Suggested threshold for ~85% recall: {chosen_threshold:.3f}, Precision: {precisions[idx]:.3f}")

    y_pred = (y_proba >= chosen_threshold).astype(int)
    print(classification_report(y, y_pred, target_names=["healthy", "failure_risk"]))

    return model, chosen_threshold


def score_all_transformers(model, df, threshold):
    proba = model.predict_proba(df[FEATURES])[:, 1]
    result = df[["transformer_id"]].copy()
    result["risk_score"] = (proba * 100).round(1)
    result["risk_tier"] = pd.cut(result["risk_score"], bins=[-0.1, 20, 50, 75, 100], labels=["low", "moderate", "elevated", "critical"])
    result["alert_flag"] = (proba >= threshold).astype(int)
    return result.sort_values("risk_score", ascending=False)


if __name__ == "__main__":
    df = load_data()
    model, threshold = train_model(df)
    scored = score_all_transformers(model, df, threshold)
    scored.to_csv("data/transformer_risk_scores.csv", index=False)
    joblib.dump(model, "models/failure_model.joblib")
    print("Saved model -> models/failure_model.joblib")
    print("Saved scores -> data/transformer_risk_scores.csv")
