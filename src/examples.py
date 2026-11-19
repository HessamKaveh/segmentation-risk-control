"""Example masks: argmax vs risk-controlled masks, on clean and blurred VOC val images.

    python src/examples.py       # -> results/figures/examples.png  (needs cache/deeplabv3.pt and cache/stats_*)

Thresholds are calibrated on all of clean VOC val (alpha = beta = 0.1), as a deployed system would do once.
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import analyze
import data
from model import build

ROOT = analyze.ROOT
dev = torch.device("cuda")


@torch.no_grad()
def probs(net, x):
    x = x[None].to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        z = net(x).float()
        z = 0.5 * (z + net(x.flip(3)).float().flip(3))
    return torch.softmax(z, 1)[0].cpu().numpy()


def overlay(img, mask, gt, colour):
    out = img.copy().astype(np.float32)
    out[mask] = 0.45 * out[mask] + 0.55 * np.array(colour, np.float32)
    cnt, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = out.astype(np.uint8).copy()
    cv2.drawContours(out, cnt, -1, (255, 255, 255), 2)
    return out


def main():
    S = analyze.Stats(os.path.join(ROOT, "cache", "stats_deeplabv3_voc_clean.npz"))
    idx = np.arange(len(S.n))
    t_crc = S.edges[analyze.crc_k(S.fnr, analyze.ALPHA)]
    t_obj = S.edges[analyze.per_object_k(S.fnr, analyze.ALPHA, analyze.BETA)]
    print(f"thresholds: crc {t_crc:.4f}  per-object {t_obj:.5f}")
    # objects of hard classes, medium size, that argmax under-segments
    hard = [data.CLASSES.index(c) for c in ["chair", "bottle", "pottedplant", "diningtable"]]
    cand = idx[np.isin(S.cls, hard) & (S.n > 4000) & (S.n < 40000) & (S.argmax_fnr > 0.3)]
    rng = np.random.default_rng(4)
    picks = rng.choice(cand, 3, replace=False)
    net = build("deeplabv3", pretrained=False).to(dev)
    net.load_state_dict(torch.load(os.path.join(ROOT, "cache", "deeplabv3.pt"), map_location=dev)); net.eval()
    rows = [(j, "clean") for j in picks] + [(picks[0], "motion_blur")]
    fig, axs = plt.subplots(len(rows), 4, figsize=(15, 3.4 * len(rows)))
    for r, (j, cond) in enumerate(rows):
        ds = data.EvalSet("voc", cond, 0 if cond == "clean" else 2)
        x, lab, _ = ds[int(S.img[j])]
        lab = np.asarray(lab); c = int(S.cls[j])
        img = np.clip((x.numpy().transpose(1, 2, 0) * data.STD + data.MEAN) * 255, 0, 255).astype(np.uint8)
        p = probs(net, x)
        gt = lab == c
        masks = [("argmax", p.argmax(0) == c, (150, 150, 150)),
                 (f"risk control (mean miss ≤ 10 %), t = {t_crc:.3f}", p[c] >= t_crc, (31, 78, 156)),
                 (f"per-object (90 % miss ≤ 10 %), t = {t_obj:.4f}", p[c] >= t_obj, (192, 57, 43))]
        axs[r, 0].imshow(img)
        axs[r, 0].set_title(f"{data.CLASSES[c]}" + (" — motion blur (sev. 2)" if cond != "clean" else ""), fontsize=10)
        for k, (name, m, col) in enumerate(masks):
            miss = 1 - (m & gt).sum() / gt.sum()
            axs[r, k + 1].imshow(overlay(img, m, gt, col))
            axs[r, k + 1].set_title(f"{name}\nmissed {100 * miss:.0f} % of the {data.CLASSES[c]}", fontsize=9)
    for a in axs.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "results", "figures", "examples.png"), dpi=110)


if __name__ == "__main__":
    main()
