import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_seg_infer_transform(img_size=256):
    """
    Segmentation inference preprocessing
    MUST match validation preprocessing used during training
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ]
    )

def preprocess_segmentation_image(image_path):
    """
    Args:
        image_path: path to input image

    Returns:
        image_tensor: [1, 3, 256, 256]
        original_shape: (H, W)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_shape = image.shape[:2]  # (H, W)

    transform = get_seg_infer_transform()
    augmented = transform(image=image)

    image_tensor = augmented["image"].unsqueeze(0)

    return image_tensor, original_shape
