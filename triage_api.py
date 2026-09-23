from contextlib import asynccontextmanager
import logging
import re
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TriageRequest(BaseModel):
    text: str

# simple safety gate (not medical advice)
EMERGENCY_PATTERNS = [
    ("chest pain", r"\bchest pain\b"),
    ("shortness of breath", r"\bshortness of breath\b"),
    ("faint", r"\bfaint(?:ing)?\b"),
    ("seizure", r"\bseizure\b"),
    ("unconscious", r"\bunconscious\b"),
]


def emergency_matches(text: str) -> List[str]:
    """Return the normalized high-risk phrases found in a message."""
    normalized = (text or "").lower()
    return [label for label, pattern in EMERGENCY_PATTERNS if re.search(pattern, normalized)]


def emergency_gate(text: str) -> bool:
    return bool(emergency_matches(text))

cat_model = None
pri_model = None

BASE_DIR = Path(__file__).resolve().parent


def _load_joblib(path: Path):
    return joblib.load(str(path))


def _get_vectorizer_and_estimator(model):
    """Best-effort extraction of (vectorizer, estimator) from a sklearn Pipeline/Calibrated model."""
    vec = None
    est = model

    # sklearn Pipeline
    if hasattr(model, "named_steps"):
        # try common names first
        for k in ("tfidf", "vectorizer", "vect", "tfidf_vectorizer"):
            if k in model.named_steps:
                vec = model.named_steps[k]
                break
        if vec is None:
            # fallback: first step with transform
            for step in model.named_steps.values():
                if hasattr(step, "transform") and hasattr(step, "fit"):
                    vec = step
                    break

        # classifier step
        for k in ("clf", "classifier", "model"):
            if k in model.named_steps:
                est = model.named_steps[k]
                break
        else:
            # fallback: last step
            try:
                est = list(model.named_steps.values())[-1]
            except Exception:
                est = model

    # Handle CalibratedClassifierCV (sklearn >=0.24)
    if hasattr(est, "calibrated_classifiers_") and getattr(est, "calibrated_classifiers_", None):
        try:
            est = est.calibrated_classifiers_[0].estimator
        except Exception:
            pass

    # Some wrappers store the underlying estimator in .estimator
    elif hasattr(est, "estimator"):
        try:
            est = est.estimator
        except Exception:
            pass

    return vec, est


def _feature_names(vec) -> Optional[np.ndarray]:
    if vec is None:
        return None
    if hasattr(vec, "get_feature_names_out"):
        return vec.get_feature_names_out()
    if hasattr(vec, "get_feature_names"):
        return np.array(vec.get_feature_names())
    return None


def _reason_phrases(model, text: str, top_k: int = 8) -> List[str]:
    """Return top contributing TF-IDF terms for the model's predicted class.

    Works for linear models with `coef_` (LogReg/LinearSVC/etc.) inside a Pipeline.
    If we can't extract weights, returns an empty list.
    """
    try:
        vec, est = _get_vectorizer_and_estimator(model)
        names = _feature_names(vec)
        if vec is None or names is None:
            return []

        # transform text
        X = vec.transform([text])
        if X is None or X.shape[1] == 0:
            return []

        # need linear coefficients
        if not hasattr(est, "coef_"):
            return []

        pred = model.predict([text])[0]

        coef = est.coef_
        # binary models may have shape (1, n_features)
        if coef.ndim == 2 and coef.shape[0] == 1:
            w = coef[0]
        else:
            # multiclass: pick row corresponding to predicted class
            classes = getattr(est, "classes_", None)
            if classes is None:
                return []
            idx = int(np.where(classes == pred)[0][0])
            w = coef[idx]

        # contributions only for terms present in the input
        # (sparse-safe): use nonzero indices
        row = X.tocsr()[0]
        if row.nnz == 0:
            return []
        idxs = row.indices
        vals = row.data

        contrib = vals * w[idxs]
        # take top positive contributors; if none positive, take top absolute
        order = np.argsort(contrib)[::-1]
        pos = [i for i in order if contrib[i] > 0]
        chosen = pos[:top_k] if len(pos) >= 1 else order[:top_k]

        phrases = [str(names[idxs[i]]) for i in chosen]

        # de-dup while preserving order
        seen = set()
        out = []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
    except Exception:
        return []

def load_models():
    global cat_model, pri_model
    try:
        cat_path = BASE_DIR / "models" / "category_model.joblib"
        pri_path = BASE_DIR / "models" / "priority_model.joblib"
        cat_model = _load_joblib(cat_path)
        pri_model = _load_joblib(pri_path)
    except Exception:
        logger.exception("Unable to load trained model artifacts")
        cat_model = None
        pri_model = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()
    yield


app = FastAPI(title="DiaCare Support Triage API", lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "ok": True,
        "api_version": "v2-ml",
        "models_loaded": bool(cat_model and pri_model),
    }

@app.post("/triage")
def triage(req: TriageRequest):
    text = (req.text or "").strip()
    matched_phrases = emergency_matches(text)

    if matched_phrases:
        return {
            "input": text,
            "priority": 5,
            "category": "medical",
            "confidence": 0.99,
            "gate_triggered": True,
            "reason_phrases": matched_phrases,
            "reason": "High-risk symptom keywords detected. Escalate immediately."
        }

    if not (cat_model and pri_model):
        return {
            "input": text,
            "error": "Models not trained/loaded. Run: python train.py",
            "api_version": "v2-ml",
            "models_loaded": False,
            "gate_triggered": False,
        }

    cat_probs = cat_model.predict_proba([text])[0]
    pri_probs = pri_model.predict_proba([text])[0]
    cat_pred = cat_model.predict([text])[0]
    pri_pred = pri_model.predict([text])[0]

    cat_phrases = _reason_phrases(cat_model, text, top_k=8)
    pri_phrases = _reason_phrases(pri_model, text, top_k=8)

    # merge and keep it short
    merged = []
    seen = set()
    for p in (cat_phrases + pri_phrases):
        if p not in seen:
            seen.add(p)
            merged.append(p)
        if len(merged) >= 10:
            break

    return {
        "input": text,
        "priority": int(pri_pred),
        "category": str(cat_pred),
        "confidence": float(max(cat_probs.max(), pri_probs.max())),
        "gate_triggered": False,
        "reason_phrases": merged,
        "reason": "Top weighted TF-IDF terms for the predicted class (linear model explainability)."
    }
