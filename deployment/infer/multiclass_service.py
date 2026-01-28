import os
import torch

from preprocess.clf_preprocess import preprocess_classification_image
from models.multiclass.multiclass_model import SkinLesionClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "multiclass", "best_multiclass_efficientnet.pth"
)

IDX_TO_CLASS = {0: "MEL", 1: "BCC", 2: "AKIEC"}

def load_model():
    model = SkinLesionClassifier(num_classes=3, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

MODEL = load_model()

def run_multiclass(image_path):
    input_tensor = preprocess_classification_image(image_path).to(device)

    with torch.no_grad():
        logits = MODEL(input_tensor)

    probs = torch.softmax(logits, dim=1).squeeze()

    probs_dict = {
        IDX_TO_CLASS[i]: probs[i].item()
        for i in range(len(IDX_TO_CLASS))
    }

    pred = max(probs_dict, key=probs_dict.get)
    return probs_dict, pred
