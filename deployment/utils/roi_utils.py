import cv2
import numpy as np

def extract_roi(image_path, mask_path, threshold=0.5):
    """
    Apply segmentation mask to original image to extract ROI

    Args:
        image_path (str): path to original RGB image
        mask_path (str): path to predicted segmentation mask
        threshold (float): binarization threshold

    Returns:
        roi (np.ndarray): masked RGB image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load mask (grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Mask not found")

    # Normalize mask to [0,1]
    mask = mask / 255.0

    # Binarize mask
    mask = (mask > threshold).astype(np.uint8)

    # Expand mask to 3 channels
    mask = np.expand_dims(mask, axis=-1)

    # Apply mask
    roi = image * mask

    return roi
