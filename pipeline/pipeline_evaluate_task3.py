import os
import csv
import cv2
import pandas as pd

from segmentation_infer import run_segmentation
from binary_infer import run_binary_inference
from multiclass_infer import run_multiclass_inference
from utils import extract_roi

# ==================================================
# CONFIGURATION
# ==================================================

# ISIC 2018 Task-3 TEST images directory
IMAGES_DIR = "../ISIC2018_Task3_Test_Images"

# ISIC 2018 Task-3 TEST ground truth CSV
GT_CSV = "../ISIC2018_Task3_Test_GroundTruth.csv"

# Output CSV
OUTPUT_CSV = "pipeline_task3_results.csv"

# Temporary files (reused each iteration)
TEMP_MASK = "temp_mask.png"
TEMP_ROI = "temp_roi.png"

# Save qualitative outputs?
SAVE_VISUALS = True
MAX_VISUALS = 50

MASK_DIR = "eval_masks"
ROI_DIR = "eval_rois"

# Benign and malignant mapping
BENIGN_CLASSES = ["NV", "BKL", "DF", "VASC"]
MALIGNANT_CLASSES = ["MEL", "BCC", "AKIEC"]

# ==================================================
# SETUP OUTPUT DIRECTORIES
# ==================================================
if SAVE_VISUALS:
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(ROI_DIR, exist_ok=True)

# ==================================================
# HELPER: One-hot → pipeline label
# ==================================================
def get_ground_truth(row):
    for cls in BENIGN_CLASSES:
        if row[cls] == 1:
            return "Benign"
    for cls in MALIGNANT_CLASSES:
        if row[cls] == 1:
            return cls
    return None

# ==================================================
# MAIN EVALUATION FUNCTION
# ==================================================
def evaluate_pipeline():

    df = pd.read_csv(GT_CSV)

    results = []
    correct_count = 0
    total_images = 0

    print("\n🚀 Starting end-to-end pipeline evaluation...\n")

    for idx, row in df.iterrows():

        image_id = row["image"]
        image_path = os.path.join(IMAGES_DIR, image_id + ".jpg")

        if not os.path.exists(image_path):
            continue

        # ---- DRY RUN SAFETY (TEMPORARY) ----
        #if total_images == 50:
        #    print("🧪 Dry run completed (50 images). Stopping.")
        #    break
        total_images += 1
        
        # -------------------------------
        # Ground truth
        # -------------------------------
        gt_label = get_ground_truth(row)

        # -------------------------------
        # Segmentation
        # -------------------------------
        run_segmentation(image_path, TEMP_MASK)

        # -------------------------------
        # ROI extraction
        # -------------------------------
        roi = extract_roi(image_path, TEMP_MASK)
        cv2.imwrite(TEMP_ROI, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

        # -------------------------------
        # Binary classification
        # -------------------------------
        prob, binary_pred = run_binary_inference(TEMP_ROI)

        # -------------------------------
        # Final pipeline decision
        # -------------------------------
        if binary_pred == "Benign":
            final_pred = "Benign"
        else:
            _, final_pred = run_multiclass_inference(TEMP_ROI)

        is_correct = (final_pred == gt_label)
        correct_count += int(is_correct)

        # -------------------------------
        # Save qualitative outputs (limited)
        # -------------------------------
        mask_path = ""
        roi_path = ""

        if SAVE_VISUALS and len(results) < MAX_VISUALS:
            mask_path = os.path.join(MASK_DIR, f"{image_id}_mask.png")
            roi_path = os.path.join(ROI_DIR, f"{image_id}_roi.png")

            cv2.imwrite(mask_path, cv2.imread(TEMP_MASK))
            cv2.imwrite(roi_path, cv2.imread(TEMP_ROI))

        # -------------------------------
        # Store result
        # -------------------------------
        results.append([
            image_id,
            gt_label,
            binary_pred,
            final_pred,
            is_correct,
            mask_path,
            roi_path
        ])

        if total_images % 50 == 0:
            print(f"Processed {total_images} images...")

    # ==================================================
    # SAVE RESULTS
    # ==================================================
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "ground_truth",
            "binary_output",
            "final_prediction",
            "correct",
            "mask_path",
            "roi_path"
        ])
        writer.writerows(results)

    accuracy = correct_count / len(results)

    print("\n==============================")
    print(f"✅ Total images evaluated : {total_images}")
    print(f"🎯 Pipeline accuracy     : {accuracy * 100:.2f}%")
    print(f"📄 Results saved to      : {OUTPUT_CSV}")
    print("==============================\n")


# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == "__main__":
    evaluate_pipeline()
