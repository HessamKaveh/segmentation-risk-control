"""Run each model on VOC val (clean + 4 corruptions x 3 severities) and on Oxford-IIIT Pet, and store compact
per-object statistics from which any threshold mask can be evaluated later without the images.

An "object" here is one (image, class) pair present in the ground truth. For every object we store histograms of the
model's probability for that class, p_c, over
  - ground-truth pixels in the interior,   - ground-truth pixels within 3 px of the object boundary,
  - all other (non-object, non-ignored) pixels,
on a fixed grid of probability bin edges (log-spaced below 0.01, linear above). A threshold mask {p_c >= t} at any
grid edge t then has an exactly known miss rate (false-negative rate) and false-positive area.

    python src/infer.py            # -> cache/stats_{model}_{condition}.npz
"""
import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import corruptions
import data
from model import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGES = np.unique(np.r_[0.0, np.geomspace(1e-6, 0.01, 300), np.linspace(0.01, 1.0, 700), 1.0 + 1e-9])
CONDITIONS = ["voc:clean:0"] + [f"voc:{c}:{s}" for c in corruptions.CORRUPTIONS for s in (1, 2, 3)] + ["pets:clean:0"]


@torch.no_grad()
def run(model, ds, dev):
    B = len(EDGES) - 1
    out = {k: [] for k in ["img", "cls", "n_gt", "n_boundary", "h_int", "h_bnd", "h_neg", "argmax_hit"]}
    conf = np.zeros((21, 21), np.int64)
    for x, lab, i in DataLoader(ds, batch_size=1, num_workers=8):
        x = x.to(dev).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x).float()
            logits = 0.5 * (logits + model(x.flip(3)).float().flip(3))      # flip test
        p = torch.softmax(logits, 1)[0].cpu().numpy()                       # (21, H, W)
        lab = lab[0].numpy()
        valid = lab != 255
        pred = p.argmax(0)
        conf += np.bincount(lab[valid].astype(np.int64) * 21 + pred[valid], minlength=441).reshape(21, 21)
        for c in np.unique(lab[valid]):
            if c == 0:
                continue
            g = lab == c
            dist = cv2.distanceTransform(g.astype(np.uint8), cv2.DIST_L2, 3)
            bnd = g & (dist <= 3)
            pc = p[c]
            out["img"].append(int(i)); out["cls"].append(int(c))
            out["n_gt"].append(int(g.sum())); out["n_boundary"].append(int(bnd.sum()))
            out["h_int"].append(np.histogram(pc[g & ~bnd], EDGES)[0].astype(np.int32))
            out["h_bnd"].append(np.histogram(pc[bnd], EDGES)[0].astype(np.int32))
            out["h_neg"].append(np.histogram(pc[valid & ~g], EDGES)[0].astype(np.int32))
            out["argmax_hit"].append(int((pred[g] == c).sum()))
    res = {k: np.array(v) for k, v in out.items()}
    res["confusion"] = conf
    res["edges"] = EDGES
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["deeplabv3", "lraspp"])
    a = ap.parse_args()
    dev = torch.device("cuda")
    for m in a.models:
        net = build(m, pretrained=False).to(dev).to(memory_format=torch.channels_last)
        net.load_state_dict(torch.load(os.path.join(ROOT, "cache", f"{m}.pt"), map_location=dev))
        net.eval()
        for cond in CONDITIONS:
            src, c, s = cond.split(":")
            name = f"{src}_{c}" + (f"_s{s}" if c != "clean" else "")
            f = os.path.join(ROOT, "cache", f"stats_{m}_{name}.npz")
            if os.path.exists(f):
                continue
            r = run(net, data.EvalSet(src, c, int(s)), dev)
            np.savez_compressed(f, **r)
            cf = r["confusion"]
            iou = np.diag(cf) / (cf.sum(0) + cf.sum(1) - np.diag(cf) + 1e-9)
            print(f"{m} {name}: {len(r['img'])} objects, mIoU {iou.mean():.3f}" if src == "voc" else
                  f"{m} {name}: {len(r['img'])} objects", flush=True)


if __name__ == "__main__":
    main()
