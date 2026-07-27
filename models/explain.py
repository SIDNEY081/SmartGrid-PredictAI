"""
Predict AI - Explainability helpers
=====================================
Shared functions for turning a trained tree model into human-readable
explanations: global feature importance (which features matter overall)
and per-instance explanations (why THIS transformer/meter got its score).

Uses SHAP's TreeExplainer when the `shap` package is importable - it's exact
and fast for tree ensembles (Random Forest / Gradient Boosting / Isolation
Forest) and needs no background dataset for the tree case. If `shap` isn't
installed, falls back to a simpler approximation built from the model's own
`feature_importances_` and how far each feature value sits from the training
population (z-score) - clearly weaker than SHAP (it ignores interactions
between features) but needs no extra dependency and still answers "which
features pushed this score up or down."
"""

import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def global_importance(model, feature_names):
    """Return a DataFrame of features ranked by the model's own global
    importance, most important first. Only works for models that expose
    `feature_importances_` (Random Forest, Gradient Boosting) - for models
    that don't (Isolation Forest), use importance_from_contributions()
    instead."""
    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def importance_from_contributions(contributions):
    """Global importance derived from per-row contributions (mean absolute
    value across all rows) instead of the model's own attributes - works for
    any model explain_batch() can produce contributions for, including
    Isolation Forest which has no `feature_importances_`."""
    return (
        contributions.abs().mean()
        .rename("importance")
        .rename_axis("feature")
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _fallback_contributions(model, X, feature_names):
    """Approximate per-row contributions when shap isn't available: global
    importance (or, if the model has none - e.g. Isolation Forest - equal
    weighting) times each feature's z-score against the batch it's being
    scored with. Positive means "pushed the score toward the flagged class",
    magnitude is only meaningful relative to other features on the same row,
    not across rows."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.ones(len(feature_names))
    mean = X.mean(axis=0)
    std = X.std(axis=0).replace(0, 1)
    z = (X - mean) / std
    return z.multiply(importances, axis=1)


def explain_batch(model, X, feature_names):
    """Return a DataFrame (same shape as X) of per-feature contributions
    for every row in X, plus a boolean `used_shap` flag on the result via
    the returned tuple (contributions_df, used_shap)."""
    X = X[feature_names]
    if _HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X)
        # Binary classifiers: shap_values returns a list [class0, class1] on
        # older shap versions, or a single (n, features, 2) array on newer
        # ones - normalize both to "contribution toward the positive class".
        if isinstance(raw, list):
            values = raw[1] if len(raw) > 1 else raw[0]
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            values = raw[:, :, 1]
        else:
            values = raw
        return pd.DataFrame(values, columns=feature_names, index=X.index), True
    return _fallback_contributions(model, X, feature_names), False


def top_reasons(contributions_row, n=3):
    """Given one row of per-feature contributions, return the top-n
    (feature, contribution) pairs by absolute magnitude, largest first."""
    ordered = contributions_row.abs().sort_values(ascending=False).head(n)
    return [(feat, contributions_row[feat]) for feat in ordered.index]


def describe_reasons(reasons):
    """Turn top_reasons() output into a plain-English sentence fragment,
    e.g. 'high age_years, low oil_quality_index'."""
    parts = []
    for feature, value in reasons:
        direction = "high" if value > 0 else "low"
        parts.append(f"{direction} {feature}")
    return ", ".join(parts)
