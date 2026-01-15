# multiclass_infer.py

import torch
import os

from multiclass_model import SkinLesionClassifier
from clf_preprocess import preprocess_classification_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping (CONFIRM WITH DATASET)
IDX_TO_CLASS = {
    0: "MEL",
    1: "BCC",
    2: "AKIEC"
}

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


def run_multiclass_inference(image_path):
    model = load_multiclass_model()

    input_tensor = preprocess_classification_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)

    probs = torch.softmax(logits, dim=1).squeeze()
    pred_idx = torch.argmax(probs).item()

    return probs.cpu().numpy(), IDX_TO_CLASS[pred_idx]



if __name__ == "__main__":
    test_image = os.path.join("..", "test_images", "sample.jpg")
    probs, label = run_multiclass_inference(test_image)

    print("=== Multiclass Classification Result ===")
    for i, p in enumerate(probs):
        print(f"{IDX_TO_CLASS[i]}: {p:.4f}")
    print(f"Prediction: {label}")
