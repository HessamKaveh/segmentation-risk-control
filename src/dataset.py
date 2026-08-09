import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

NUM_CLASSES = 21  # Pascal VOC


class PascalVOCSegmentation(Dataset):
    """Pascal VOC segmentation dataset"""
    def __init__(self, train=True):
        self.dataset = datasets.VOCSegmentation(
            root=DATA_DIR, year='2012', image_set='train' if train else 'val',
            download=True, transforms=None
        )
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        
        img = self.img_transform(img)
        
        mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask_array = np.array(mask)
        mask_tensor = torch.from_numpy(mask_array).long()

        return img, mask_tensor


def get_dataloaders(batch_size=16):
    train_set = PascalVOCSegmentation(train=True)
    val_set = PascalVOCSegmentation(train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
