import torch
import os

from clf_preprocess import preprocess_classification_image
from step7_model import EfficientNetB0Binary

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load Binary Model ONCE
# --------------------------------------------------
def load_binary_model():
    model = EfficientNetB0Binary(
        pretrained=False,   # IMPORTANT during inference
        num_classes=1
    )

    weights_path = os.path.join(
        "..", "weights", "binary", "best_efficientnet_b0_binary.pth"
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


# Load model globally (CRITICAL FIX)
MODEL = load_binary_model()


# --------------------------------------------------
# Inference
# --------------------------------------------------
def run_binary_inference(image_path):
    input_tensor = preprocess_classification_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = MODEL(input_tensor)

    prob = torch.sigmoid(logits).item()
    label = "Malignant" if prob >= 0.5 else "Benign"

    return prob, label


# --------------------------------------------------
# Standalone test (optional)
# --------------------------------------------------
if __name__ == "__main__":

    image_path = os.path.join("..", "test_images", "sample.jpg")

    prob, label = run_binary_inference(image_path)

    print("=== Binary Classification Result ===")
    print(f"Malignant probability : {prob:.4f}")
    print(f"Prediction            : {label}")
