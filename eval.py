import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.u2net import U2NETP
from datasets.transforms import get_val_transforms

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH_SIZE = 4

BASE_DIR = "/content/drive/MyDrive/segmentation_project"
IMG_DIR = f"{BASE_DIR}/data/images"
MASK_DIR = f"{BASE_DIR}/data/masks"
SPLIT_DIR = f"{BASE_DIR}/splits"
OUT_DIR = f"{BASE_DIR}/outputs"
VIS_DIR = f"{OUT_DIR}/eval_vis"

os.makedirs(VIS_DIR, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
class ISICDataset(Dataset):
    def __init__(self, split_file, transforms=None):
        with open(split_file) as f:
            self.images = f.read().splitlines()
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(IMG_DIR, img_name)
        mask_path = os.path.join(
            MASK_DIR, img_name.replace(".jpg", "_segmentation.png")
        )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            out = self.transforms(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]

        mask = (mask > 0).float().unsqueeze(0)
        return image, mask, img_name

# -------------------------
# Metrics
# -------------------------
def dice_coef(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)

def iou_coef(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

# -------------------------
# Evaluation
# -------------------------
def evaluate():
    ds = ISICDataset(
        f"{SPLIT_DIR}/val.txt",
        transforms=get_val_transforms(IMG_SIZE)
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = U2NETP().to(DEVICE)
    model.load_state_dict(
        torch.load(f"{OUT_DIR}/best_u2netp.pth", map_location=DEVICE)
    )
    model.eval()

    dices, ious = [], []

    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            preds_bin = (preds > 0.5).float()

            for i in range(preds_bin.size(0)):
                d = dice_coef(preds_bin[i], masks[i]).item()
                j = iou_coef(preds_bin[i], masks[i]).item()
                dices.append(d)
                ious.append(j)

                # save a few visualizations
                if len(dices) <= 10:
                    p = preds_bin[i].squeeze().cpu().numpy() * 255
                    gt = masks[i].squeeze().cpu().numpy() * 255
                    cv2.imwrite(
                        os.path.join(VIS_DIR, names[i].replace(".jpg", "_pred.png")),
                        p.astype(np.uint8)
                    )
                    cv2.imwrite(
                        os.path.join(VIS_DIR, names[i].replace(".jpg", "_gt.png")),
                        gt.astype(np.uint8)
                    )

    print(f"📊 Mean Dice: {np.mean(dices):.4f}")
    print(f"📊 Mean IoU : {np.mean(ious):.4f}")

if __name__ == "__main__":
    evaluate()
