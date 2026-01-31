import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm

from models.u2net import U2NETP
from datasets.transforms import get_train_transforms, get_val_transforms

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4

BASE_DIR = "/content/drive/MyDrive/segmentation_project"
IMG_DIR = f"{BASE_DIR}/data/images"
MASK_DIR = f"{BASE_DIR}/data/masks"
SPLIT_DIR = f"{BASE_DIR}/splits"
OUT_DIR = f"{BASE_DIR}/outputs"

os.makedirs(OUT_DIR, exist_ok=True)

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
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float() / 255.0
        return image, mask

# -------------------------
# Losses
# -------------------------
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, smooth=1.0):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (
            preds.sum() + targets.sum() + smooth
        )
        return 1 - dice

# -------------------------
# Training
# -------------------------
def train():
    train_ds = ISICDataset(
        f"{SPLIT_DIR}/train.txt",
        transforms=get_train_transforms(IMG_SIZE)
    )
    val_ds = ISICDataset(
        f"{SPLIT_DIR}/val.txt",
        transforms=get_val_transforms(IMG_SIZE)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = U2NETP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    dice = DiceLoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = bce(preds, masks) + dice(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(imgs)
                loss = bce(preds, masks) + dice(preds, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                f"{OUT_DIR}/best_u2netp.pth"
            )
            print("✅ Saved best model")

if __name__ == "__main__":
    train()
