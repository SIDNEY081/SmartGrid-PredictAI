# SmartGrid PredictAI

Synthetic-data prototype for Eskom's transformer failure prediction and
electricity theft detection platform.

## Structure

```
SmartGrid-PredictAI/
  data/
    generate_data.py               # builds the synthetic transformer/meter datasets below
    transformer_data.csv           # 800 transformers, ~17% failure_within_1yr rate
    meter_data.csv                 # 3,000 meters, ~11% is_theft rate
    transformer_failure_scores.csv # notebook output: failure_score + tier per transformer
    meter_anomaly_scores.csv       # notebook output: anomaly_score + tier per meter
    transformer_risk_scores.csv    # script output: risk_score + tier per transformer
    meter_theft_scores.csv         # script output: anomaly_score + tier per meter
  notebooks/
    transformer_failure_model.ipynb  # Random Forest classifier -> failure_score
    theft_detection_model.ipynb      # Isolation Forest anomaly detector -> anomaly_score
  models/
    failure_prediction.py          # script version of the failure model
    theft_detection.py             # script version of the theft model
  requirements.txt
```

## Notebooks vs. scripts

Both pipelines are implemented twice: once as notebooks (`notebooks/`) and
once as standalone scripts (`models/*.py`). They use the same features and
model families but are independent implementations, and they write to
**different** output files (`transformer_failure_scores.csv`/
`meter_anomaly_scores.csv` for the notebooks vs. `transformer_risk_scores.csv`/
`meter_theft_scores.csv` for the scripts) so running one doesn't clobber the
other's results.

**Evaluation methodology differs, and it matters:** `models/failure_prediction.py`
evaluates on a held-out test split (ROC-AUC ≈ 0.77 on the current synthetic
data). `notebooks/transformer_failure_model.ipynb` evaluates in-sample (trains
and scores the same rows), which lets the Random Forest memorize the data and
report a misleading ROC-AUC of 1.0. Trust the script's number, not the
notebook's, for the failure model. (The theft/anomaly side doesn't have this
problem — Isolation Forest is unsupervised and both implementations report a
consistent ROC-AUC ≈ 0.98 against the synthetic `is_theft` labels.)

## Why these model choices

**Failure prediction — Random Forest Classifier (supervised)**
Eskom has labeled history here: past transformers that did fail, with sensor
readings leading up to the event. That makes this a supervised problem.
Random Forest handles mixed-scale tabular features well and gives a
probability score that converts directly into a risk tier.

**Theft detection — Isolation Forest (unsupervised)**
Confirmed theft cases are rare and biased (you only find what you go and
check), so training a classifier purely on confirmed cases would just learn
"where we've looked before." Isolation Forest instead flags statistical
outliers in consumption behavior without needing labels, which is closer to
how a real deployment would work. The strongest engineered feature is the
ratio between what the transformer fed to a connection and what the meter
declared — a classic real-world signature of bypass or tampering.

## How to run

Regenerate the synthetic data (run from the repo root):

```bash
python3 data/generate_data.py   # -> data/transformer_data.csv, data/meter_data.csv
```

Notebooks — open and run all cells in:

```bash
notebooks/transformer_failure_model.ipynb   # trains + scores -> data/transformer_failure_scores.csv
notebooks/theft_detection_model.ipynb       # trains + scores -> data/meter_anomaly_scores.csv
```

Both notebooks read their input CSV from `../data/...`, so run them from
inside the `notebooks/` working directory (Jupyter does this by default).

Scripts — run from the repo root:

```bash
python3 models/failure_prediction.py   # trains + scores -> data/transformer_risk_scores.csv, models/failure_model.joblib
python3 models/theft_detection.py      # trains + scores -> data/meter_theft_scores.csv, models/theft_model.joblib
```

## Next steps

1. Give `notebooks/transformer_failure_model.ipynb` a held-out train/test
   split (matching `models/failure_prediction.py`) so its reported ROC-AUC
   stops being inflated by in-sample evaluation.
2. Replace the synthetic CSVs in `data/` with real Eskom data once available,
   matching the same column schema (or update `generate_data.py` /
   `FEATURES` in the models if the real schema differs).
3. Add a feedback loop: investigation outcomes (confirmed theft / false
   positive, confirmed failure / false alarm) should flow back in to improve
   precision over time.
