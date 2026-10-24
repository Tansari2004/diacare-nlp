# Experiment configuration — grouped v2

- Dataset: 630 synthetic support messages; 84 reviewed intent families.
- Shared split: first fold of five-fold StratifiedGroupKFold, joint category/priority strata, shuffle enabled, seed 42. No seed/fold search.
- Training: 490 rows / 67 families. Holdout: 140 rows / 17 families (22.2%).
- Group and normalized-text overlap: zero, asserted before fitting.
- Features: TF-IDF unigrams/bigrams, up to 20,000 features.
- Models: linear SVM; balanced priority class weights; seed 42.
- Calibration: sigmoid on three family-disjoint training folds, wrapping complete TF-IDF/SVM pipelines. Vocabulary fitted inside each fold. Confidence reliability is not independently established.
- Evaluation: individual category/priority accuracy, macro F1 and confusion matrices; both-correct accuracy; complete HTTP API with keyword overrides on the same rows.
- No five-fold performance CV score is reported for this new experiment. The three folds are for fitting calibration only.
- Artifacts: annotations, source hashes, split/calibration manifest, row-level predictions, full metrics, saved models.
- No tuning after observing this holdout. A fresh final test set is needed for later model selection.

The previous separate row-level random-split evaluation is archived as legacy, not directly comparable to this changed split/calibration protocol.
