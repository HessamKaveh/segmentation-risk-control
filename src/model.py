"""Two segmentation networks at the two ends of the speed/accuracy range a robot might deploy.

deeplabv3  DeepLabV3, ResNet-50 backbone (ImageNet-initialised), output stride 16
lraspp     LR-ASPP, MobileNetV3-Large backbone (ImageNet-initialised), a real-time mobile model
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
                      replace_stride_with_dilation=[False, False, True])
        self.backbone = IntermediateLayerGetter(bb, {"layer4": "out"})
        self.head = DeepLabHead(2048, num_classes)

    def forward(self, x):
        h, w = x.shape[-2:]
        return F.interpolate(self.head(self.backbone(x)["out"]), (h, w), mode="bilinear", align_corners=False)


class LRASPP(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        self.net = lraspp_mobilenet_v3_large(
            weights=None, num_classes=num_classes,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)

    def forward(self, x):
        return self.net(x)["out"]


def build(name, pretrained=True):
    return {"deeplabv3": DeepLabV3, "lraspp": LRASPP}[name](pretrained=pretrained)
