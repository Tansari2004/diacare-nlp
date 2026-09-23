# Experiment configuration

- **Features:** TF-IDF unigrams and bigrams, maximum 20,000 features
- **Model:** Linear SVM with sigmoid probability calibration
- **Priority weighting:** Balanced class weights
- **Cross-validation:** Five-fold stratified CV on the training pool
- **Holdout:** Stratified 20% test set, untouched during training
- **Primary metric:** Macro F1
- **Additional metrics:** Accuracy, per-class precision/recall/F1 and confusion matrix
- **Random seed:** 42
- **Safety layer:** Deterministic emergency-keyword override before model inference

The dataset is synthetic and domain-aligned. See [`data/DATASET_INFO.md`](data/DATASET_INFO.md) for its dataset card.
