from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42

import numpy as np
np.random.seed(SEED)

def train_classifier(df, y_col):
    X = df["text"].astype(str).fillna("")
    y = df[y_col].astype(str)

    counts = Counter(y)
    can_stratify = (y.nunique() > 1) and (min(counts.values()) >= 2)

    # --- HOLDOUT SPLIT (never touched during training) ---
    X_train_pool, X_holdout, y_train_pool, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y if can_stratify else None,
    )

    vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1,
        max_features=20000,
        strip_accents="unicode"
    )
    if y_col == "priority":
        base = LinearSVC(class_weight="balanced")
    else:
        base = LinearSVC()

    min_class = min(counts.values()) if len(counts) else 0
    if min_class >= 3:
        cv = 3
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    elif min_class >= 2:
        cv = 2
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    else:
        clf = base

    pipe = Pipeline([("tfidf", vec), ("clf", clf)])

    # --- 5-FOLD CROSS VALIDATION ON TRAIN POOL ---
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import numpy as np

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        pipe,
        X_train_pool,
        y_train_pool,
        cv=cv,
        scoring="f1_macro"
    )

    print(f"\n=== {y_col.upper()} CROSS-VALIDATION ===")
    print(f"Macro F1 mean: {cv_scores.mean():.3f}")
    print(f"Macro F1 std:  {cv_scores.std():.3f}")

    # --- FINAL TRAINING ---
    pipe.fit(X_train_pool, y_train_pool)

    # --- HOLDOUT EVALUATION ---
    y_pred = pipe.predict(X_holdout)

    # --- ERROR ANALYSIS: PRINT MISCLASSIFIED EXAMPLES ---
    misclassified = []
    for text, true, pred in zip(X_holdout, y_holdout, y_pred):
        if true != pred:
            misclassified.append((text, true, pred))

    print(f"\nTop 15 misclassified examples for {y_col}:")
    for ex in misclassified[:15]:
        print(f"TEXT: {ex[0]}")
        print(f"TRUE: {ex[1]}  ->  PRED: {ex[2]}")
        print("---")

    print(f"\n=== {y_col.upper()} HOLDOUT TEST REPORT ===")
    print(classification_report(y_holdout, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_holdout, y_pred))

    return pipe
def main():
    import pandas as pd
    df = pd.read_csv("data/messages.csv")

    print("\nTRAINING CATEGORY MODEL...")
    train_classifier(df, "category")

    print("\nTRAINING PRIORITY MODEL...")
    train_classifier(df, "priority")

if __name__ == "__main__":
    main()