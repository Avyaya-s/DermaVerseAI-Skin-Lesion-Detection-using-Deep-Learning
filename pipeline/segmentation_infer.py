import torch
import os
import cv2
import numpy as np

from seg_preprocess import preprocess_segmentation_image
from u2net import U2NETP

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load U2NETP model ONCE
# --------------------------------------------------
def load_u2netp():
    model = U2NETP(in_ch=3, out_ch=1)

    weights_path = os.path.join(
        "..", "weights", "segmentation", "best_u2netp.pth"
    )

    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


# Load model globally (CRITICAL FIX)
MODEL = load_u2netp()


# --------------------------------------------------
# Inference
# --------------------------------------------------
def run_segmentation(image_path, output_mask_path):

    input_tensor, original_shape = preprocess_segmentation_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        mask = MODEL(input_tensor)

    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255

    cv2.imwrite(output_mask_path, mask)
    return output_mask_path


# --------------------------------------------------
# Standalone test (optional)
# --------------------------------------------------
if __name__ == "__main__":

    image_path = os.path.join("..", "test_images", "sample.jpg")
    output_mask = "segmentation_mask.png"

    run_segmentation(image_path, output_mask)

    print("✅ Segmentation mask saved as segmentation_mask.png")
