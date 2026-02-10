import torch
import torch.nn as nn
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os
# -------------------------
# Configuration
# -------------------------


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "label_map.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.pth")

EMBEDDER_NAME = "all-MiniLM-L6-v2"   # will auto-download locally
CONFIDENCE_THRESHOLD = 0.60          # adjustable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load label map
# -------------------------
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

num_classes = len(label_map)

# -------------------------
# Classifier definition (must match training)
# -------------------------


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------
# Load model
# -------------------------
model = IntentClassifier(384, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------------------------
# Load sentence embedder
# -------------------------
embedder = SentenceTransformer(EMBEDDER_NAME)

# -------------------------
# Intent prediction function
# -------------------------


def predict_intent(text: str):
    # Generate embedding
    embedding = embedder.encode([text], convert_to_numpy=True)
    embedding = torch.tensor(embedding, dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(embedding)
        probs = torch.softmax(logits, dim=1)

    confidence, pred_id = torch.max(probs, dim=1)
    confidence = confidence.item()
    pred_id = pred_id.item()

    predicted_intent = label_map[pred_id]

    # Unknown handling
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "intent": "unknown",
            "confidence": confidence
        }

    return {
        "intent": predicted_intent,
        "confidence": confidence
    }


# -------------------------
# Test manually
# -------------------------
if __name__ == "__main__":
    while True:
        text = input("\nUser: ")
        if text.lower() in ["exit", "quit"]:
            break

        result = predict_intent(text)
        print("Predicted intent:", result["intent"])
        print("Confidence:", round(result["confidence"], 3))
