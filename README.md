# Predict AI - Technical Prototype

Synthetic-data prototype for Eskom's transformer failure prediction and
electricity theft detection platform.

## Structure

```
predict_ai/
  data/
    generate_synthetic_data.py     # generates realistic synthetic datasets
    transformer_data.csv           # 9,600 rows: 800 transformers x 12 months
    meter_data.csv                 # 36,000 rows: 3,000 meters x 12 months
    transformer_risk_scores.csv    # model output: 0-100 risk score per row
    meter_anomaly_scores.csv       # model output: 0-100 anomaly score per row
  models/
    failure_prediction_model.py    # Gradient Boosting classifier
    theft_detection_model.py       # Isolation Forest anomaly detector
    failure_model.joblib           # trained model artifact
    theft_model.joblib             # trained model artifact
```

## Why these model choices

**Failure prediction - Gradient Boosting Classifier (supervised)**
Eskom has labeled history here: past transformers that did fail, with sensor
readings leading up to the event. That makes this a supervised problem.
Gradient boosting (or XGBoost/LightGBM in a full production build) handles
mixed-scale tabular features well, is fast to retrain monthly, and produces
well-calibrated probabilities that convert directly into a risk score.

**Theft detection - Isolation Forest (unsupervised)**
Confirmed theft cases are rare and biased (you only find what you go and
check), so training a classifier purely on confirmed cases would just learn
"where we've looked before." Isolation Forest instead flags statistical
outliers in consumption behavior without needing labels, which is closer to
how a real deployment would work. The single most powerful engineered
feature is the ratio between what the transformer fed to a connection and
what the meter declared - a classic real-world signature of bypass or
tampering.

## Results on synthetic data

| Model | Metric | Score |
|---|---|---|
| Failure prediction | ROC-AUC | 0.68 |
| Failure prediction | PR-AUC (vs. 25% base rate) | 0.39 |
| Theft detection | ROC-AUC (vs. synthetic ground truth) | 0.95 |
| Theft detection | PR-AUC (vs. 8% base rate) | 0.42 |

These numbers are from fully synthetic data and will look different (likely
better, once tuned) on real SCADA/smart-meter data - the point of this
prototype is to demonstrate the pipeline and modeling approach, not to claim
production-ready accuracy.

## How to run

```bash
cd predict_ai
python3 data/generate_synthetic_data.py       # regenerate datasets
python3 models/failure_prediction_model.py    # train + score transformers
python3 models/theft_detection_model.py       # train + score meters
```

## Next steps for a real deployment

1. Replace synthetic generators with real feeds: SCADA historian exports,
   DGA lab results, AMI/smart meter reads, GIS feeder topology.
2. Retrain failure model monthly; retrain/recalibrate theft model
   quarterly as consumption patterns and tariffs shift seasonally.
3. Add a feedback loop: investigation outcomes (confirmed theft / false
   positive) should flow back in to improve precision over time
   (semi-supervised refinement of the Isolation Forest, or a second-stage
   classifier trained on confirmed cases only).
4. Push `risk_score` / `anomaly_score` outputs into a data warehouse table
   that Power BI reads directly
