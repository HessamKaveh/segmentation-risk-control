"""Train on VOC 2012 train_aug (10,582 images).  python src/train.py --model deeplabv3 --iters 30000

SGD with poly learning-rate decay, 384x384 random crops, random scale 0.5-2.0, horizontal flip, mixed precision.
The last iteration is kept (no checkpoint selection on val).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
from model import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {"deeplabv3": dict(batch=8, lr=0.01), "lraspp": dict(batch=16, lr=0.02)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CFG))
    ap.add_argument("--iters", type=int, default=30000)
    a = ap.parse_args()
    torch.manual_seed(0)
    cfg = CFG[a.model]
    dev = torch.device("cuda")
    model = build(a.model).to(dev).to(memory_format=torch.channels_last)
    params = [{"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": cfg["lr"]},
              {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": cfg["lr"] * 10}]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (1 - it / a.iters) ** 0.9)
    scaler = torch.amp.GradScaler()
    loader = DataLoader(data.VOCTrain(), batch_size=cfg["batch"], shuffle=True, drop_last=True, num_workers=12,
                        persistent_workers=True, pin_memory=True)
    it, t0, hist, run = 0, time.time(), [], []
    while it < a.iters:
        for x, y in loader:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x).float(), y, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run.append(loss.item()); it += 1
            if it % 500 == 0:
                hist.append((it, float(np.mean(run)))); run = []
                eta = (time.time() - t0) / it * (a.iters - it) / 60
                print(f"{a.model} it {it}/{a.iters} loss {hist[-1][1]:.3f} ({(time.time() - t0) / 60:.0f} min, "
                      f"eta {eta:.0f} min)", flush=True)
            if it >= a.iters:
                break
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "history"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ROOT, "cache", f"{a.model}.pt"))
    json.dump(hist, open(os.path.join(ROOT, "results", "history", f"{a.model}.json"), "w"))


if __name__ == "__main__":
    main()
