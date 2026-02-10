import os
import random
import json
import torch
import nltk

from nltk_utils import bag_of_words, tokenize
from model import NeuralNet

nltk.download('punkt')

# -------- PATHS (MUST COME FIRST) --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(BASE_DIR, "intents.json")
DATA_FILE = os.path.join(BASE_DIR, "data.pth")
# ----------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(INTENTS_FILE, 'r') as json_data:
    intents = json.load(json_data)

data = torch.load(DATA_FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])

    return "I do not understand..."
