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
# Load U2NETP model
# --------------------------------------------------
def load_u2netp():
    model = U2NETP(in_ch=3, out_ch=1)

    weights_path = os.path.join(
        "..", "weights", "segmentation", "best_u2netp.pth"
    )

    checkpoint = torch.load(weights_path, map_location=device)

    # Handle checkpoint dictionary vs raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

# --------------------------------------------------
# Inference
# --------------------------------------------------
if __name__ == "__main__":

    model = load_u2netp()

    image_path = os.path.join("..", "test_images", "sample.jpg")

    input_tensor, original_shape = preprocess_segmentation_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        mask = model(input_tensor)  # sigmoid already applied

    mask = mask.squeeze().cpu().numpy()  # [256, 256]

    # Resize mask back to original image resolution
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

    # Binarize mask
    mask = (mask > 0.5).astype(np.uint8) * 255

    cv2.imwrite("segmentation_mask.png", mask)

    print("✅ Segmentation mask saved as segmentation_mask.png")
