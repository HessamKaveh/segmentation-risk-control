"""PASCAL VOC 2012 (+ SBD 'train_aug' labels) and Oxford-IIIT Pet.

train : the 10,582 'train_aug' images (VOC 2012 train + SBD), none of them in VOC 2012 val
val   : the 1,449 VOC 2012 val images, later split by image into calibration / test halves
pets  : the 3,669 Oxford-IIIT Pet test images (cats and dogs); trimap 'pet' -> VOC class cat (8) or dog (12),
        trimap 'border' ignored. Used only as a dataset-shift test, never for training.
"""
import os
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import corruptions

VOC = os.environ.get("VOC_DIR", os.path.expanduser("~/datasets/portfolio/voc"))
PETS = os.environ.get("PETS_DIR", os.path.expanduser("~/datasets/portfolio/pets"))
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def voc_ids(split):
    seg = os.path.join(VOC, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation")
    val = [l.strip() for l in open(os.path.join(seg, "val.txt")) if l.strip()]
    if split == "val":
        return val
    aug = sorted(f[:-4] for f in os.listdir(os.path.join(VOC, "SegmentationClassAug")) if f.endswith(".png"))
    return sorted(set(aug) - set(val))


def to_tensor(img_rgb):
    return torch.from_numpy(((img_rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())


class VOCTrain(Dataset):
    def __init__(self, crop=384):
        self.ids, self.crop = voc_ids("train"), crop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img = cv2.cvtColor(cv2.imread(os.path.join(VOC, "VOCdevkit", "VOC2012", "JPEGImages", self.ids[i] + ".jpg")),
                           cv2.COLOR_BGR2RGB)
        lab = cv2.imread(os.path.join(VOC, "SegmentationClassAug", self.ids[i] + ".png"), cv2.IMREAD_GRAYSCALE)
        s = rng.uniform(0.5, 2.0)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        c = self.crop
        ph, pw = max(c - img.shape[0], 0), max(c - img.shape[1], 0)
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(124, 116, 104))
            lab = cv2.copyMakeBorder(lab, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
        y, x = rng.integers(0, img.shape[0] - c + 1), rng.integers(0, img.shape[1] - c + 1)
        img, lab = img[y:y + c, x:x + c], lab[y:y + c, x:x + c]
        if rng.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        return to_tensor(img), torch.from_numpy(lab.astype(np.int64).copy())


class EvalSet(Dataset):
    """Full-resolution evaluation images (VOC val or Pets test), optionally corrupted (seeded per image)."""

    def __init__(self, source="voc", corruption="clean", severity=0):
        self.source, self.corruption, self.severity = source, corruption, severity
        if source == "voc":
            self.ids = voc_ids("val")
        else:
            self.ids = [l.split()[0] for l in open(os.path.join(PETS, "annotations", "test.txt")) if l.strip()]

    def __len__(self):
        return len(self.ids)

    def label(self, i):
        if self.source == "voc":
            return cv2.imread(os.path.join(VOC, "SegmentationClassAug", self.ids[i] + ".png"), cv2.IMREAD_GRAYSCALE)
        t = cv2.imread(os.path.join(PETS, "annotations", "trimaps", self.ids[i] + ".png"), cv2.IMREAD_GRAYSCALE)
        cls = 8 if self.ids[i][0].isupper() else 12          # Oxford-IIIT Pet: capitalised breed = cat
        lab = np.full(t.shape, 255, np.uint8)
        lab[t == 1] = cls
        lab[t == 2] = 0
        return lab

    def __getitem__(self, i):
        d = "VOCdevkit/VOC2012/JPEGImages" if self.source == "voc" else "images"
        root = VOC if self.source == "voc" else PETS
        img = cv2.imread(os.path.join(root, d, self.ids[i] + ".jpg"), cv2.IMREAD_COLOR)
        if self.source == "pets" and max(img.shape[:2]) > 500:  # VOC-like resolution
            f = 500 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        seed = zlib.crc32(f"{self.source}-{self.ids[i]}-{self.corruption}-{self.severity}".encode())
        img = corruptions.apply(img, self.corruption, self.severity, seed)
        lab = self.label(i)
        if lab.shape != img.shape[:2]:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), torch.from_numpy(lab), i
