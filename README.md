# SmartGrid PredictAI

Synthetic-data prototype for Eskom's transformer failure prediction,
electricity theft detection, and feeder outage forecasting.

## Structure

```
SmartGrid-PredictAI/
  data/
    generate_data.py               # builds the synthetic transformer/meter/feeder datasets below
    transformer_data.csv           # 3,000 transformers (each on a feeder), ~19% failure_within_1yr rate
    meter_data.csv                 # 10,000 meters, ~10% is_theft rate
    feeder_data.csv                # 1,500 feeders, ~13% outage_within_7_days rate
    transformer_failure_scores.csv # notebook output: failure_score + tier per transformer
    meter_anomaly_scores.csv       # notebook output: anomaly_score + tier per meter
    transformer_risk_scores.csv    # script output: risk_score + tier per transformer
    meter_theft_scores.csv         # script output: anomaly_score + tier per meter
    feeder_outage_scores.csv       # script output: outage_risk_score + tier per feeder
  notebooks/
    transformer_failure_model.ipynb  # Random Forest classifier -> failure_score
    theft_detection_model.ipynb      # Isolation Forest anomaly detector -> anomaly_score
  models/
    failure_prediction.py          # script version of the failure model
    theft_detection.py             # script version of the theft model
    outage_forecasting.py          # feeder outage model (script only, no notebook - see below)
    explain.py                     # shared SHAP / feature-importance helpers used by all three
  dashboard/
    app.py                         # Flask app serving the planner dashboard
    chatbot.py                     # rule-based Q&A over the score CSVs (no LLM)
    streamlit_app.py               # interactive upload-CSV / Predict console (see below)
    templates/index.html
    static/style.css
  tests/                           # pytest smoke tests for the schema + pipelines
  requirements.txt
```

## Notebooks vs. scripts

The failure and theft pipelines are each implemented twice: once as notebooks
(`notebooks/`) and once as standalone scripts (`models/*.py`). They use the
same features and model families but are independent implementations, and
they write to **different** output files (`transformer_failure_scores.csv`/
`meter_anomaly_scores.csv` for the notebooks vs. `transformer_risk_scores.csv`/
`meter_theft_scores.csv` for the scripts) so running one doesn't clobber the
other's results.

**Outage forecasting is script-only, deliberately** — it doesn't have a
notebook twin. Duplicating an implementation is exactly what caused most of
the schema-drift and inflated-metric bugs found earlier in this project, so
the third model isn't repeating that pattern.

Both failure-prediction implementations now evaluate on a held-out test split
before refitting on the full dataset for deployment scoring (ROC-AUC ≈ 0.86
in both the script and the notebook). The theft/anomaly side evaluates
in-sample in both implementations, which is fine there since Isolation Forest
is unsupervised and can't memorize labels it never sees; both report a
consistent ROC-AUC ≈ 0.98 against the synthetic `is_theft` labels. Outage
forecasting evaluates on a held-out split too (ROC-AUC ≈ 0.75) - weaker than
the other two, which tracks: feeder outages depend on real-world factors
(weather, vegetation growth, crew response time) this synthetic dataset
doesn't model, so treat it as "clearly better than chance," not a precise
estimate.

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

**Outage forecasting — Gradient Boosting Classifier (supervised)**
Feeder-level outage history is Eskom's own labeled record, same reasoning as
failure prediction. The interesting part is the feature set: alongside a
feeder's own condition (age, vegetation encroachment, protection equipment
age, load, historical outages), it includes `critical_transformer_count` —
how many transformers on that feeder the failure model already flagged
`critical` — pulled from `data/transformer_risk_scores.csv`. That's the one
place these three models actually talk to each other: a feeder with several
already-risky transformers on it is a more plausible near-term outage than
one judged purely on its own attributes. Run `models/failure_prediction.py`
before `models/outage_forecasting.py` for this reason. It uses the same
model capacity as the failure model (`n_estimators=200, max_depth=3`) - an
earlier, much smaller version of `data/feeder_data.csv` (200 rows) needed
shallower trees to avoid saturating predicted probabilities toward 0/1, but
that stopped being true once the feeder count grew.

## How to run

Regenerate the synthetic data (run from the repo root):

```bash
python3 data/generate_data.py   # -> data/transformer_data.csv, data/meter_data.csv, data/feeder_data.csv
```

Notebooks — open and run all cells in:

```bash
notebooks/transformer_failure_model.ipynb   # trains + scores -> data/transformer_failure_scores.csv
notebooks/theft_detection_model.ipynb       # trains + scores -> data/meter_anomaly_scores.csv
```

Both notebooks read their input CSV from `../data/...`, so run them from
inside the `notebooks/` working directory (Jupyter does this by default).

Scripts — run from the repo root, **in this order** (outage forecasting
reads failure prediction's output):

```bash
python3 models/failure_prediction.py   # trains + scores -> data/transformer_risk_scores.csv, models/failure_model.joblib
python3 models/theft_detection.py      # trains + scores -> data/meter_theft_scores.csv, models/theft_model.joblib
python3 models/outage_forecasting.py   # trains + scores -> data/feeder_outage_scores.csv, models/outage_model.joblib
```

Dashboard — a planner-facing view of the script outputs (tier breakdown,
flagged counts, top-15 highest-risk lists) for all three models:

```bash
python3 dashboard/app.py   # -> http://127.0.0.1:5000
```

Run the scripts first (or at least once) so `transformer_risk_scores.csv`,
`meter_theft_scores.csv`, and `feeder_outage_scores.csv` exist — the
dashboard just reads them, it doesn't train anything itself. It needs
internet access once to load Plotly from a CDN.

**Streamlit app** — a second, more visual console covering all three models
(the Flask dashboard above is the lighter-weight one):

```bash
streamlit run dashboard/streamlit_app.py   # -> http://localhost:8501
```

It auto-loads and scores `data/transformer_data.csv`, `data/meter_data.csv`,
and `data/feeder_data.csv` on open — no CSV upload step. It prefers the
already-trained `models/*.joblib` and pre-scored `data/*_scores.csv` files
(run the scripts above first for instant load); if those don't exist yet it
trains on the spot and writes them, so it also works from a clean checkout.
A "Refresh predictions" button in the sidebar re-reads everything (use it
after re-running a script). Each of the three tabs (Transformer Failure /
Theft Detection / Feeder Outage) shows a severity-tier breakdown, the top-15
riskiest rows with their reason text, a global feature-importance chart, and
a SHAP "why" chart for any single row you pick from the top 30.

**Chat** — both apps have the same rule-based (no LLM) Q&A over the score
CSVs: a floating widget in the Flask dashboard, a sidebar panel in the
Streamlit app. Answers things like "how many feeders are critical?" or
"what's the risk score for T0208?" by keyword-matching the question onto a
pandas filter — no API key, no external network call, no cost. See
`dashboard/chatbot.py` (shared by both apps).

**Power BI / Tableau**: `transformer_risk_scores.csv`, `meter_theft_scores.csv`,
and `feeder_outage_scores.csv` are meant to be imported directly (Get Data ->
Text/CSV) - no server needed. Each has a numeric `*_tier_order` column
(1=low … 4=critical) alongside the text tier column, so BI tools can sort by
severity instead of alphabetically. `transformer_risk_scores.csv` and
`feeder_outage_scores.csv` share a `feeder_id` column if you want to relate
them in the data model.

## Explainable AI

All three models print held-out accuracy, precision/recall/F1
(`classification_report`), and a confusion matrix when trained
(`python3 models/failure_prediction.py`, etc.), plus a global feature
importance ranking. `models/explain.py` provides the shared logic:

- Uses `shap.TreeExplainer` when `shap` is installed (exact for tree
  ensembles, and it's the library all three models use — Random Forest,
  Gradient Boosting, Isolation Forest). Every scored row in
  `data/*_scores.csv` gets a `top_reasons` column, e.g. `"high age_years,
  low oil_quality_index, high load_factor"`.
- Falls back to a z-score-weighted approximation if `shap` isn't installed,
  so the pipelines and tests still run without it - clearly weaker (it
  ignores feature interactions) but dependency-free.

The Streamlit app (`dashboard/streamlit_app.py`) surfaces this
interactively: a bar chart of global feature importance, and a per-row SHAP
bar chart for any single transformer or meter you pick.

## Tests

```bash
python3 -m pytest tests/   # run from the repo root
```

Covers: `data/*.csv` still has every column the models expect,
`generate_data.py`'s schema stays in sync with `FEATURES`/`NUMERIC_FEATURES`
in `models/*.py`, all three pipelines run end-to-end without errors, the
failure/theft scripts' output filenames don't collide with the notebooks',
and each capacity-based alert flag actually flags close to
`CAPACITY_FRACTION` of the fleet. These tests would have caught every
schema-drift bug found earlier in this project's history.

## Next steps

1. Replace the synthetic CSVs in `data/` with real Eskom data once available,
   matching the same column schema (or update `generate_data.py` /
   `FEATURES` in the models if the real schema differs).
2. Add a feedback loop: investigation outcomes (confirmed theft / false
   positive, confirmed failure / false alarm) should flow back in to improve
   precision over time.
3. `models/failure_prediction.py` and `models/outage_forecasting.py` both
   flag the riskiest 15% (`CAPACITY_FRACTION`) instead of everything clearing
   a recall-tuned probability threshold — for the failure model that had
   been ~60% of the fleet, not a usable work list. 15% is still a guess for
   both; replace once real weekly maintenance/dispatch capacity is known.
4. Outage forecasting's ROC-AUC (~0.75) is the weakest of the three models
   and likely stays that way even with more synthetic feeders - the gap is
   about missing real-world signal (weather, live vegetation growth, crew
   response times), not sample size.
