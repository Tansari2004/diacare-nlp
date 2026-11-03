# DiaCare Support Message Classification

[![CI](https://github.com/Tansari2004/diacare-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/Tansari2004/diacare-nlp/actions/workflows/ci.yml)

A standalone prototype combining TF-IDF/linear SVM classifiers, a keyword override, and FastAPI. The current `/classify` endpoint suggests a support category and leaves every priority decision to a person. The older `/triage` priority experiment is retained as a deprecated research endpoint. It is **not a medical device or a validated clinical triage service**.

## Current scope: category suggestion, manual priority

The weak priority results below led to a narrower interface, rather than more tuning on the observed holdout. `POST /classify` returns a category suggestion and a nullable category decision; **priority is always null and must be assigned by a person**. A fixed 0.60 model-score threshold sends uncertain category decisions to `needs_review`. This threshold is an engineering default, not a validated probability or a threshold optimized on the holdout. Even returned category suggestions require human review.

On the same already observed 140-message synthetic holdout:

| Category comparison | Correct / total | Accuracy |
| --- | ---: | ---: |
| Majority training category | 17 / 140 | 12.1% |
| TF-IDF + Multinomial Naive Bayes | 104 / 140 | 74.3% |
| Existing calibrated SVM, all suggestions | 108 / 140 | 77.1% |

With the fixed review threshold, 98/140 messages receive a category suggestion (70% coverage): **83/98 are correct, with 15 errors**. The other 42 need category review; **all 140 need priority review**. The 84.7% accuracy among returned category decisions excludes abstentions and must never replace the 77.1% all-message accuracy. These checks do not establish savings in actual reviewer time or reliable clinical routing. Baselines were fitted on the same 490 training messages only; no parameters, rules or split were chosen from test scores.

Run `python evaluate_category.py` after `python train.py`. Inspect [category metrics](results/category_only_metrics.json) and [per-message responses](results/category_only_predictions.csv). The old holdout is a retrospective diagnostic, not a fresh final test. A [next-evaluation protocol](data/NEXT_EVALUATION.md) records the remaining independent review and fresh-data requirements.

## Corrected evaluation

The previous row-level evaluation had seven training-overlap messages in each of two separate holdouts. Those preliminary scores are superseded and archived in [`results/legacy_random_split_metrics.json`](results/legacy_random_split_metrics.json).

The new experiment groups duplicates and related message templates into **84 reviewed message families**. One fixed shared split reserves **140 messages from 17 families**; training uses 490 messages from the other 67 families. No annotated family or normalized text crosses the boundary. The split was fixed before new model scores were inspected, with no seed/fold search. Grouped calibration also fits each complete TF-IDF/SVM pipeline on training families only.

| System | Category accuracy | Priority accuracy | Both answers correct |
| --- | ---: | ---: | ---: |
| Models only | 108/140 · 77.1% | 77/140 · 55.0% | 63/140 · 45.0% |
| Complete HTTP API, including keyword override | 108/140 · 77.1% | 79/140 · 56.4% | 65/140 · 46.4% |

Model-only macro F1 is **0.766 for category** and **0.333 for priority**. The priority model correctly labels **0 of 11 priority-5 messages**; the keyword override raises this to **2 of 11** in the API. These results do not support reliable handling of urgent requests. The two overrides repair two joint prediction errors on this split, but that is not evidence that the rule covers unfamiliar urgent language.

The corrected evaluation changes both the split and calibration protocol. The old and new scores are not a controlled estimate of the effect of duplicates alone. No model tuning was performed on the new holdout.

### Inspect the evidence

- [Grouping protocol and limitations](data/GROUPING_PROTOCOL.md)
- [All row-level family annotations](data/message_groups.csv)
- [Frozen split and calibration-fold manifest](results/split_manifest.csv)
- [Holdout predictions, labels, and API rule matches](results/holdout_predictions.csv)
- [Full metrics and confusion matrices](results/metrics.json)
- [Evaluation summary](results/evaluation_summary.md)

Family annotations were prepared with coding-assistant support and have not had independent human or clinical review. They are conservative but not proof that all paraphrases have been identified. The 630 messages are synthetic, balanced across categories, and include no patient records. This one family-held-out experiment does not establish generalization to real support traffic. A second independent grouping review and representative external validation remain necessary.

## Training and API

Each classifier learns its TF-IDF vocabulary within its training folds, preventing vocabulary leakage into the held-out folds. Sigmoid calibration wraps the full TF-IDF/SVM pipeline, using three family-disjoint folds of the training pool. Priority uses balanced class weights. Calibration does **not** by itself establish reliable confidence estimates; probability reliability has not been independently evaluated.

**Deprecated experiment:** `POST /triage` first checks the existing keyword patterns. A match returns `prediction_source: "keyword_rule"`, explicit `rule_matches`, category `medical`, and priority 5. **Both confidence fields are null**, because no model probability was computed. The phrase list is narrow and does not understand negation or all symptom paraphrases.

Otherwise, the models return separate `category_confidence` and `priority_confidence` values. A confident category cannot conceal an uncertain priority. These are model estimates, not probabilities of clinical safety. Indicative TF-IDF terms come from one member of each calibrated ensemble and are not a complete ensemble explanation.

**Response migration:** the former single `confidence` field and fixed 0.99 rule confidence have been removed. Clients should inspect `prediction_source`, `rule_matches`, and the two nullable confidence fields.

```json
{
  "category": "medical",
  "priority": 5,
  "category_confidence": null,
  "priority_confidence": null,
  "prediction_source": "keyword_rule",
  "rule_matches": ["chest pain", "faint"],
  "gate_triggered": true
}
```

- `GET /health`: process liveness, with `models_loaded` reported separately.
- `GET /ready`: **503** when either model is unavailable; **200** when both are loaded.
- `POST /triage`: **503** if models are unavailable, including keyword paths; **422** for blank or over-5,000-character messages.
- Docker health checks use `/ready`.

## Reproduce

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python evaluate_category.py
pytest -q
uvicorn triage_api:app --reload
```

Training validates annotation hashes, writes the split manifest before fitting, saves both models, and evaluates every shared holdout message through the actual FastAPI route with TestClient. It publishes model-only and complete-API outcomes, including joint accuracy. Models remain fitted to training data only so saved artifacts reproduce the held-out experiment.

## Next experiment

Keep this holdout fixed as an observed experiment. Before selecting improvements from its errors, collect or reserve a fresh final test set. Review group assignments independently, add representative language and rare priorities, compare simple baselines, and test negation and uncertainty handling. Have qualified reviewers assess routing labels and failure consequences before proposing real-world use.

Code is licensed under MIT. See [dataset notes](data/DATASET_INFO.md).
