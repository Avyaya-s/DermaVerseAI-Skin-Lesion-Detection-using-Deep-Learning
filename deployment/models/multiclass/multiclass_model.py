# multiclass_model.py

import torch
import torch.nn as nn
import timm

class SkinLesionClassifier(nn.Module):
    """
    EfficientNet-B0 based multiclass classifier
    """
    def __init__(self, num_classes=3, pretrained=False):
        super(SkinLesionClassifier, self).__init__()

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained
        )

        num_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
