# pipeline/pipeline_inference.py

import os
import cv2

from segmentation_infer import run_segmentation
from binary_infer import run_binary_inference
from multiclass_infer import run_multiclass_inference
from utils import extract_roi


# -----------------------------
# CONFIG
# -----------------------------
TEST_IMAGE_PATH = os.path.join("..", "test_images", "ISIC_0024306.jpg")

TEMP_MASK_PATH = "temp_mask.png"
TEMP_ROI_PATH = "temp_roi.png"


def run_pipeline(image_path):
    print("\n" + "=" * 60)
    print("🧠 Skin Lesion Detection Pipeline Started")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Segmentation
    # --------------------------------------------------
    print("\n[1] Running lesion segmentation...")
    run_segmentation(image_path, TEMP_MASK_PATH)

    # --------------------------------------------------
    # 2. ROI Extraction
    # --------------------------------------------------
    print("[2] Extracting ROI using segmentation mask...")
    roi = extract_roi(image_path, TEMP_MASK_PATH)

    # Save ROI temporarily for classifier input
    cv2.imwrite(TEMP_ROI_PATH, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    # --------------------------------------------------
    # 3. Binary Classification
    # --------------------------------------------------
    print("[3] Running binary classification (Benign vs Malignant)...")
    prob, binary_label = run_binary_inference(TEMP_ROI_PATH)

    print(f"    → Malignant probability: {prob:.4f}")
    print(f"    → Binary decision      : {binary_label}")

    # --------------------------------------------------
    # 4. Conditional Multiclass Classification
    # --------------------------------------------------
    if binary_label == "Malignant":
        print("\n[4] Malignant detected → running multiclass classification...")
        probs, disease_label = run_multiclass_inference(TEMP_ROI_PATH)

        print("\n    Multiclass probabilities:")
        for cls, p in zip(["MEL", "BCC", "AKIEC"], probs):
            print(f"      {cls}: {p:.4f}")

        print(f"\n    ✅ Final diagnosis: {disease_label}")

    else:
        print("\n    ✅ Final diagnosis: Benign lesion (no further classification required)")

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline(TEST_IMAGE_PATH)
