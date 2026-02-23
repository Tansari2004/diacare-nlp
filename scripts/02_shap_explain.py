import os
import shap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

def predict_proba(texts):
    # SHAP may pass a numpy array or other sequence type; normalize to list[str]
    if not isinstance(texts, list):
        texts = list(texts)
    texts = [str(t) for t in texts]

    toks = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**toks).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

def main():
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker, output_names=["NEGATIVE", "POSITIVE"])

    texts = [
        "Diacare helped me understand what to do next.",
        "I’m confused and the app doesn’t explain my numbers.",
        "The meal plan feature is actually useful.",
        "I would not recommend this app to anyone.",
    ]

    sv = explainer(texts)

    html = shap.plots.text(sv, display=False)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/shap_text.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Wrote outputs/shap_text.html")

if __name__ == "__main__":
    main()
