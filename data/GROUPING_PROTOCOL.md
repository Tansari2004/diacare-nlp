# Grouped evaluation protocol

The original experiment used independent row-level random splits for category and priority. Each holdout contained seven normalized texts also present in its training pool. Its scores are archived in `results/legacy_random_split_metrics.json` and must not be presented as the current evaluation.

## Annotation before fitting

All 630 source rows were read and assigned to **84 broad message families**, recorded in `message_groups.csv` before the new models were fitted. The file records row positions, original text, raw-text SHA-256, and a family name. Duplicate messages and recurring request templates stay together. Examples include rescheduling/cancellation requests, double charges, unavailable premium access, night-time lows, dose questions, and data exports. Several families cross category boundaries, such as appointment display, units, notifications, exports, and missing booking confirmations.

These are manually reviewed support-intent groupings, not expert clinical annotations. Some combined symptoms and related requests are grouped conservatively to avoid splitting closely related scenarios. This is not a guarantee that every semantic similarity or paraphrase has been identified; a second independent grouping review is still desirable. The group file was fixed before new model scores were inspected. It is not adjusted to improve scores.

## Fixed shared holdout

Use the **first fold** from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, with joint category/priority labels for approximate balance and the reviewed family as the group. No seed or fold search. This produces **490 training rows from 67 families and 140 holdout rows from 17 families**. Group sizes mean the holdout is 22.2% rather than exactly 20%. Both classifiers and the complete HTTP API use these same 140 messages.

The train/test family intersection and normalized-text intersection must both be empty. Text normalization applies Unicode NFKC, case folding, and punctuation/whitespace normalization. Raw-text hashes also make annotation/source drift fail explicitly.

## Training, calibration and evaluation

Three family-disjoint folds of the training pool are used for sigmoid calibration. The complete TF-IDF/SVM pipeline is fitted inside each calibration training fold, including its vocabulary. These folds are not another reported performance experiment. No holdout information is used to fit vocabulary, SVMs, or calibration.

`results/split_manifest.csv` records every row's family, split, hash, and calibration validation-fold assignment. `results/holdout_predictions.csv` records truth labels, model-only predictions, both API predictions, rule matches, and separate confidence values. `results/metrics.json` includes individual category/priority accuracy and macro F1, both-correct accuracy, and the corresponding complete-API results using real `POST /triage` requests through FastAPI TestClient.

A corrected split does not establish real-world generalization or reliable probabilities. It is one synthetic family-held-out experiment. Further changes selected after seeing this holdout would need a fresh final test set.
