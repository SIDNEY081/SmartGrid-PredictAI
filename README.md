# SmartGrid PredictAI

Synthetic-data prototype for Eskom's transformer failure prediction and
electricity theft detection platform.

## Structure

```
SmartGrid-PredictAI/
  data/
    transformer_data.csv           # sample transformer readings + failure_within_1yr label
    meter_data.csv                 # sample smart-meter readings + is_theft label
    transformer_failure_scores.csv # model output: failure_score + tier per transformer
    meter_anomaly_scores.csv       # model output: anomaly_score + tier per meter
  notebooks/
    transformer_failure_model.ipynb  # Random Forest classifier -> failure_score
    theft_detection_model.ipynb      # Isolation Forest anomaly detector -> anomaly_score
  models/
    failure_prediction.py          # draft script version of the failure model (schema not yet
                                    # in sync with data/transformer_data.csv - see note below)
    theft_detection.py             # draft script version of the theft model (schema not yet
                                    # in sync with data/meter_data.csv - see note below)
  requirements.txt
```

## Current source of truth

The **notebooks** are the working, up-to-date pipeline — they're what actually
produced `transformer_failure_scores.csv` and `meter_anomaly_scores.csv`.

The `.py` files under `models/` were an earlier attempt to turn the notebooks
into standalone scripts, but their feature lists (e.g.
`capacity_kva`, `dissolved_gas_ppm`, `area_id`, `customer_type`) don't match
the columns actually present in `data/transformer_data.csv` /
`data/meter_data.csv` yet, so running them as-is will fail. Treat them as a
refactor-in-progress rather than the current pipeline.

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

Open and run all cells in:

```bash
notebooks/transformer_failure_model.ipynb   # trains + scores -> data/transformer_failure_scores.csv
notebooks/theft_detection_model.ipynb       # trains + scores -> data/meter_anomaly_scores.csv
```

Both notebooks read their input CSV from `../data/...`, so run them from
inside the `notebooks/` working directory (Jupyter does this by default).

## Next steps

1. Bring `models/failure_prediction.py` and `models/theft_detection.py` in
   line with the notebooks so there's a script-based pipeline for
   automation/CI, not just notebooks.
2. Replace the small sample CSVs in `data/` with larger synthetic (or real)
   datasets once the schema is finalized.
3. Add a feedback loop: investigation outcomes (confirmed theft / false
   positive, confirmed failure / false alarm) should flow back in to improve
   precision over time.
