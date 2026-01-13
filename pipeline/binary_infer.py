import torch
import os
import cv2

from clf_preprocess import preprocess_classification_image
from step7_model import EfficientNetB0Binary   # <-- import YOUR model

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load Binary Model
# --------------------------------------------------
def load_binary_model():
    model = EfficientNetB0Binary(
        pretrained=False,   # IMPORTANT: False during inference
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

# --------------------------------------------------
# Inference
# --------------------------------------------------
if __name__ == "__main__":

    model = load_binary_model()

    image_path = os.path.join("..", "test_images", "sample.jpg")

    input_tensor = preprocess_classification_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)

    # inspect raw data output, raw logits
    
    # print("Raw logits:", logits.item()) 


    prob = torch.sigmoid(logits).item()
    label = "Malignant" if prob >= 0.5 else "Benign"

    print("=== Binary Classification Result ===")
    print(f"Malignant probability : {prob:.4f}")
    print(f"Prediction            : {label}")
