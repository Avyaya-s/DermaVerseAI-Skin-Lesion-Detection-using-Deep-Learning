import cv2
import numpy as np

def extract_roi(image_path, mask_path):
    """
    Applies segmentation mask to original image

    Returns:
        roi_image (RGB)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)

    roi = image * mask
    roi = roi.astype(np.uint8)

    return roi
