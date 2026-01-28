import os
import torch

from preprocess.clf_preprocess import preprocess_classification_image
from models.binary.step7_model import EfficientNetB0Binary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "binary", "best_efficientnet_b0_binary.pth"
)

def load_model():
    model = EfficientNetB0Binary(pretrained=False, num_classes=1)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

MODEL = load_model()

def run_binary(image_path):
    input_tensor = preprocess_classification_image(image_path).to(device)

    with torch.no_grad():
        logits = MODEL(input_tensor)

    prob = torch.sigmoid(logits).item()
    label = "Malignant" if prob >= 0.5 else "Benign"

    return prob, label
