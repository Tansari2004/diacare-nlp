import torch
from transformers import pipeline

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    device = 0 if torch.cuda.is_available() else -1  # (CUDA rarely on Mac)
    clf = pipeline("sentiment-analysis", model=MODEL_ID)

    samples = [
        "Diacare helped me understand what to do next.",
        "I’m confused and the app doesn’t explain my numbers.",
        "The meal plan feature is actually useful.",
    ]

    for s in samples:
        out = clf(s)[0]
        print(f"text: {s}")
        print(f"  label={out['label']} score={out['score']:.4f}")

if __name__ == "__main__":
    main()
