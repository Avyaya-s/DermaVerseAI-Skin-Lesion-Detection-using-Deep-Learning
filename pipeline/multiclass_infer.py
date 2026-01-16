import torch
import os

from clf_preprocess import preprocess_classification_image
from multiclass_model import SkinLesionClassifier  # adjust filename if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX_TO_CLASS = {
    0: "MEL",
    1: "BCC",
    2: "AKIEC"
}

# --------------------------------------------------
# Load model ONCE
# --------------------------------------------------
def load_multiclass_model():
    model = SkinLesionClassifier(
        num_classes=3,
        pretrained=False
    )

    weights_path = os.path.join(
        "..", "weights", "multiclass", "best_multiclass_efficientnet.pth"
    )

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# Load model globally (IMPORTANT)
MODEL = load_multiclass_model()


# --------------------------------------------------
# Inference
# --------------------------------------------------
def run_multiclass_inference(image_path):
    input_tensor = preprocess_classification_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = MODEL(input_tensor)

    probs = torch.softmax(logits, dim=1).squeeze()

    probs_dict = {
        IDX_TO_CLASS[i]: probs[i].item()
        for i in range(len(IDX_TO_CLASS))
    }

    pred_label = max(probs_dict, key=probs_dict.get)

    return probs_dict, pred_label
