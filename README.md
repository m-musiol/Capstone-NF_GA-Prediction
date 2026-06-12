![Logo](pics/logo.png)

# Google Analytics Customer Revenue Prediction

A modern, reproducible Python pipeline for the Kaggle
[GStore Customer Revenue Prediction](https://www.kaggle.com/c/ga-customer-revenue-prediction)
competition. The task: for each visitor of the Google Merchandise Store, predict
the **natural log of their total future revenue** over a forward-looking window.

> This repository started as my 2019 capstone at the neuefische Data Science
> bootcamp. It has since been rebuilt from a set of hand-written notebooks into a
> packaged, tested pipeline. The original notebooks are preserved under
> [`legacy/`](legacy/) for reference. See
> [What changed](#what-changed-vs-the-2019-version) for the details.

---

## The problem in one paragraph

For most businesses a tiny fraction of customers drives the majority of revenue.
Given ~1.7 million Google Analytics sessions, we must predict how much money each
visitor will spend in a future period. Almost every visitor spends **nothing**, so
the target is extremely sparse — fewer than ~1.5% of visitors ever convert. Naively
regressing revenue wastes all the model's capacity on the zeros.

## Methodology: a two-stage model

The pipeline uses the competition-proven **two-stage** structure, which separates
the two questions hidden in the task:

1. **Classification** — a LightGBM classifier estimates `P(visitor returns and buys)`.
2. **Regression** — a LightGBM regressor, trained *only on actual buyers*, estimates
   the log-revenue conditional on a purchase happening.

The expected log-revenue used for scoring is the product of the two:

```
predicted_log_revenue = P(buyer) × conditional_log_revenue
```

### Forward-looking timeframes

Sessions are aggregated **per visitor** over a *feature window*, then labelled with
the revenue they generate in a later, disjoint *prediction window* (with a gap in
between, mirroring the competition framing of 168 / 46 / 62 days). This is what
makes the model genuinely predictive rather than descriptive. See
[`features/timeframe.py`](src/garev/features/timeframe.py).

---

## Project structure

```
src/garev/
├── config.py            # typed, frozen configuration for every stage
├── data/
│   ├── schema.py        # single source of truth for the GA export layout
│   ├── sample.py        # synthetic data generator (no 31 GB download needed)
│   ├── loader.py        # JSON flattening + type coercion
│   └── clean.py         # constant-column drop + structural missing-value fills
├── features/
│   ├── encoders.py      # fit-once categorical encoding (fixes the train/test bug)
│   └── timeframe.py     # per-visitor aggregation + forward-looking target
├── models/
│   ├── two_stage.py     # the classifier + regressor estimator
│   ├── tune.py          # optional Optuna hyper-parameter search
│   └── evaluate.py      # AUC / average precision / RMSE + feature importance
├── pipeline.py          # raw CSV → trained model → submission
└── cli.py               # `garev` command-line interface
tests/                   # full suite, runs end to end on synthetic data
legacy/                  # the original 2019 notebooks, untouched
```

---

## Quick start

Requires **Python 3.9+** (3.11+ recommended).

```bash
# 1. Install the package and its dev tools
pip install -e ".[dev]"

# 2. Generate a small synthetic dataset (no Kaggle download required)
garev generate-sample --out data/sample.csv --visitors 2000

# 3. Train and evaluate
garev evaluate --data data/sample.csv

# 4. Train, persist the model, and write a submission file
garev train   --data data/sample.csv --out artifacts/model.joblib
garev predict --data data/sample.csv --model artifacts/model.joblib --out submission.csv
```

Or use it as a library:

```python
from garev import load_sessions, make_timeframe, train, predict_submission

sessions   = load_sessions("data/sample.csv")
dataset    = make_timeframe(sessions)
trained    = train(dataset)
submission = predict_submission(trained, sessions)
```

### Optional extras

```bash
pip install -e ".[tune]"     # Optuna hyper-parameter search
pip install -e ".[explain]"  # SHAP + matplotlib for explainability
```

---

## Using the real Kaggle data

The real `train_v2.csv` (24 GB) and `test_v2.csv` (7 GB) are **not** included.
Download them from the
[competition data page](https://www.kaggle.com/c/ga-customer-revenue-prediction/data)
into `data/`, then point the same commands at them:

```bash
garev train   --data data/train_v2.csv --out artifacts/model.joblib
garev predict --data data/test_v2.csv  --model artifacts/model.joblib --out submission.csv
```

The loader reads the file in chunks, so the 24 GB file does not need to fit in RAM.

---

## What changed vs the 2019 version

The original notebooks worked but were beginner-grade and have aged poorly. The
rewrite fixes correctness bugs and modernises the engineering:

**Correctness bugs fixed**

- `LabelEncoder` was fitted *separately* on train and test, giving the same
  category different codes in each split. Replaced by a single fit-once
  `OrdinalEncoder` that also handles unseen categories gracefully.
- A copy-paste bug wrote test data into a train column
  (`df_train[...] = pd.to_datetime(df_test[...])`).
- `lgb.LightGBM()` (a non-existent class) would have crashed the grid search.
- The model-comparison notebook scored almost every model with another model's
  predictions (`model_knn.predict` everywhere).

**Modernisation**

- A brittle hand-rolled recursive JSON parser (with chained `str.replace` calls)
  replaced by `json.loads` + `pandas.json_normalize`.
- Deprecated APIs removed (`pandas.io.json.json_normalize`,
  `infer_datetime_format`, `from plotly.offline import *`, blanket
  `warnings.filterwarnings("ignore")`).
- ~10 intermediate CSV round-trips replaced by an in-memory pipeline.
- Notebooks refactored into a tested `src/` package with typed config, a CLI,
  early stopping, proper cross-window evaluation, optional Optuna tuning, and a
  synthetic data generator so everything runs without the 31 GB download.

---

## Development

```bash
pytest            # run the test suite (uses synthetic data, no download)
ruff check .      # lint
ruff format .     # format
mypy src          # type-check
```

## License

MIT — see [LICENSE](LICENSE).
