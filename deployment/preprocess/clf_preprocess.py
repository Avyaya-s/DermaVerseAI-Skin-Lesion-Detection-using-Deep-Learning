# pipeline/clf_preprocess.py
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

def preprocess_classification_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = val_transform(image=image)['image']
    return image.unsqueeze(0)  # shape: [1, 3, 224, 224]
