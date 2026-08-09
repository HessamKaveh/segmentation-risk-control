import torch
import torch.nn as nn
import torchvision.models as models


class SimpleFCN(nn.Module):
    """Fully Convolutional Network ساده برای Semantic Segmentation"""
    def __init__(self, num_classes=21):
        super().__init__()
        
        # Encoder: VGG16
        vgg16 = models.vgg16(weights="DEFAULT")
        self.features = vgg16.features
        
        # Freeze encoder
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Classifier (جایگزین FC layers)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        # Encoder
        h = self.features(x)  # (B, 512, H/32, W/32)
        
        # Classifier
        h = self.classifier(h)  # (B, num_classes, H/32, W/32)
        
        # Upsample back to input size
        h = self.upsample(h)  # (B, num_classes, H, W)
        
        return h
