import os
import cv2
import torch
import numpy as np

from preprocess.seg_preprocess import preprocess_segmentation_image
from models.segmentation.u2net import U2NETP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "segmentation", "best_u2netp.pth"
)

def load_model():
    model = U2NETP(in_ch=3, out_ch=1)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

MODEL = load_model()

def run_segmentation(image_path, save_path):
    input_tensor, original_shape = preprocess_segmentation_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        mask = MODEL(input_tensor)

    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255

    cv2.imwrite(save_path, mask)
    return save_path
