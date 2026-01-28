import os
import cv2

from infer.segmentation_service import run_segmentation
from infer.binary_service import run_binary
from infer.multiclass_service import run_multiclass
from utils.roi_utils import extract_roi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_MASK = os.path.join(BASE_DIR, "temp", "mask.png")
TEMP_ROI = os.path.join(BASE_DIR, "temp", "roi.png")

os.makedirs(os.path.dirname(TEMP_MASK), exist_ok=True)

def run_full_pipeline(image_path):

    # 1. Segmentation
    run_segmentation(image_path, TEMP_MASK)

    # 2. ROI extraction
    roi = extract_roi(image_path, TEMP_MASK)
    cv2.imwrite(TEMP_ROI, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    # 3. Binary classification
    prob, binary_label = run_binary(TEMP_ROI)

    if binary_label == "Benign":
        return {
            "binary_prediction": "Benign",
            "binary_probability": prob,
            "final_prediction": "Benign",
            "multiclass_probabilities": None
        }

    # 4. Multiclass classification
    probs, final_label = run_multiclass(TEMP_ROI)

    return {
        "binary_prediction": "Malignant",
        "binary_probability": prob,
        "final_prediction": final_label,
        "multiclass_probabilities": probs
    }
