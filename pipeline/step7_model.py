# Step7_model.py
"""
Define EfficientNet-B0 model for binary classification
"""

import torch
import torch.nn as nn
import timm

class EfficientNetB0Binary(nn.Module):
    """
    EfficientNet-B0 for binary classification (Benign vs Malignant)
    Uses pretrained weights from ImageNet
    """
    def __init__(self, pretrained=True, num_classes=1):
        """
        Args:
            pretrained: Use ImageNet pretrained weights
            num_classes: 1 for binary classification with BCEWithLogitsLoss
        """
        super(EfficientNetB0Binary, self).__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )

        # Get number of features from backbone
        self.num_features = self.backbone.num_features

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 3, 224, 224]

        Returns:
            logits: Output tensor [batch_size, 1] for binary classification
        """
        # Extract features
        features = self.backbone(x)  # [batch_size, num_features, 7, 7]

        # Global pooling
        pooled = self.global_pool(features)  # [batch_size, num_features, 1, 1]
        pooled = pooled.flatten(1)  # [batch_size, num_features]

        # Classification
        logits = self.classifier(pooled)  # [batch_size, 1]

        return logits


def create_model(pretrained=True, device='cuda'):
    """
    Create and initialize the model
    """
    model = EfficientNetB0Binary(pretrained=pretrained, num_classes=1)
    model = model.to(device)

    print("=" * 60)
    print("🤖 MODEL ARCHITECTURE")
    print("=" * 60)
    print(f"Model: EfficientNet-B0")
    print(f"Pretrained: {pretrained}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Device: {device}")

    return model


print("✅ Model architecture defined!")