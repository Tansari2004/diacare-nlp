# Corrected shared-holdout evaluation

630 synthetic rows; 84 reviewed families. Fixed shared holdout: 140 messages from 17 families; training: 490 rows from 67 families. Zero normalized-text or annotated-family overlap. See [grouping protocol](../data/GROUPING_PROTOCOL.md).

| System | Category correct | Priority correct | Both correct |
| --- | ---: | ---: | ---: |
| Models only | 108/140 (77.1%) | 77/140 (55.0%) | 63/140 (45.0%) |
| Complete HTTP API | 108/140 (77.1%) | 79/140 (56.4%) | 65/140 (46.4%) |

Model-only macro F1: category 0.766; priority 0.333.

The priority model correctly classifies 0 of 11 priority-5 examples; the API correctly classifies 2 of 11. The narrow keyword override fires twice, fixing two joint errors and breaking no previously correct joint prediction on this particular holdout. No model or keyword changes were selected from these errors. These outcomes do not support dependable urgent-message handling.

Three family-disjoint training folds fit sigmoid calibration of complete TF-IDF/SVM pipelines. These are fitting folds, not independent performance validation or proof of probability reliability.

The previous 92.9% category / 80.2% priority figures used separate row-level holdouts with duplicate overlap and are archived in [legacy metrics](legacy_random_split_metrics.json). The new split and calibration protocol both differ, so the change is not attributable to duplicates alone.

Inspect [all holdout predictions](holdout_predictions.csv), the [split manifest](split_manifest.csv), and [full metrics](metrics.json). One manually grouped synthetic holdout does not establish real-world generalization; additional independent grouping and representative testing remain necessary.
