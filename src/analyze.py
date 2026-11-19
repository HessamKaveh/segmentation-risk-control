"""Risk-controlled segmentation masks: all experiments and figures.

    python src/analyze.py          # needs cache/stats_*.npz from src/infer.py -> results/results.json, figures

Objects = (image, class) pairs in the ground truth. Calibration / test = random 50/50 split of VOC val by image,
repeated 200 times; brackets are 2.5-97.5 % ranges over the repeats.

Masks are {pixels with p_c >= t}. Methods choose t:
  argmax       the usual segmentation output (pixel goes to its most likely class); no threshold, no guarantee
  crc          conformal risk control (Angelopoulos et al., 2022): largest t such that
               n/(n+1) * mean_cal FNR(t) + 1/(n+1) <= alpha   ->   E[FNR] <= alpha on new objects
  crc-class    the same, separately for each of the 20 classes (class-conditional risk)
  per-object   split conformal on t_i = the largest t at which object i still has FNR <= beta, taking the
               floor((n+1) alpha)-th smallest t_i   ->   P(FNR <= beta) >= 1 - alpha for each new object
FNR = share of the object's pixels missed by the mask.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import corruptions
from data import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE, RES = os.path.join(ROOT, "cache"), os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
ALPHA, BETA = 0.10, 0.10
REPEATS = int(os.environ.get("REPEATS", 200))
MODELS = {"deeplabv3": "DeepLabV3-ResNet50", "lraspp": "LR-ASPP-MobileNetV3"}
METHODS = ["argmax", "crc", "crc-class", "per-object"]
MLABEL = {"argmax": "argmax mask (standard)", "crc": "risk control (mean miss <= 10 %)",
          "crc-class": "risk control per class", "per-object": "per-object (90 % of objects miss <= 10 %)"}


class Stats:
    def __init__(self, f):
        z = np.load(f)
        self.img, self.cls, self.n = z["img"], z["cls"], z["n_gt"].astype(np.float64)
        gt = z["h_int"] + z["h_bnd"]
        z0 = np.zeros((len(self.n), 1))
        self.miss = np.hstack([z0, np.cumsum(gt, 1)])                         # missed pixels at threshold k
        self.miss_bnd = np.hstack([z0, np.cumsum(z["h_bnd"], 1)])
        self.n_bnd = z["n_boundary"].astype(np.float64)
        neg = z["h_neg"]
        self.fp = neg.sum(1, keepdims=True) - np.hstack([z0, np.cumsum(neg, 1)])
        self.fnr = self.miss / self.n[:, None]                                # (objects, K) non-decreasing in k
        self.argmax_fnr = 1 - z["argmax_hit"] / self.n
        self.confusion, self.edges = z["confusion"], z["edges"]
        self.K = self.fnr.shape[1]

    def miou(self):
        cf = self.confusion
        return float((np.diag(cf) / (cf.sum(0) + cf.sum(1) - np.diag(cf) + 1e-9)).mean())


def crc_k(fnr, alpha):
    n = len(fnr)
    if n == 0:
        return 0
    risk = n / (n + 1) * fnr.mean(0) + 1 / (n + 1)
    ok = np.where(risk <= alpha)[0]
    return int(ok.max()) if len(ok) else 0


def per_object_k(fnr, alpha, beta):
    t = (fnr <= beta).sum(1) - 1                 # largest k with FNR <= beta (fnr is monotone in k)
    n = len(t)
    j = int(np.floor((n + 1) * alpha))
    return 0 if j < 1 else int(np.sort(t)[j - 1])


def fit(S, idx, method, alpha=ALPHA, beta=BETA):
    if method == "argmax":
        return None
    if method == "crc":
        return crc_k(S.fnr[idx], alpha)
    if method == "per-object":
        return per_object_k(S.fnr[idx], alpha, beta)
    return {c: crc_k(S.fnr[idx][S.cls[idx] == c], alpha) for c in range(1, 21)}


def apply(S, idx, method, k):
    """Per-object FNR, mask area / object area, boundary FNR on the objects idx."""
    if method == "argmax":
        return S.argmax_fnr[idx], np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    kk = np.array([k[c] for c in S.cls[idx]]) if isinstance(k, dict) else np.full(len(idx), k)
    r = np.arange(len(idx))
    fnr = S.fnr[idx][r, kk]
    area = (S.n[idx] - S.miss[idx][r, kk] + S.fp[idx][r, kk]) / S.n[idx]
    bnd = S.miss_bnd[idx][r, kk] / np.maximum(S.n_bnd[idx], 1)
    return fnr, area, bnd


def splits(S, seed):
    imgs = np.unique(S.img)
    cal_imgs = np.random.default_rng(seed).permutation(imgs)[: len(imgs) // 2]
    cal = np.isin(S.img, cal_imgs)
    return np.where(cal)[0], np.where(~cal)[0]


def agg(vals):
    v = np.array(vals)
    return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}


def evaluate(S_cal, S_test, method, groups=None, same=True, alpha=ALPHA, repeats=None):
    """Calibrate on S_cal's calibration half, test on S_test's test half (same split, or all of S_test)."""
    out = {k: [] for k in ["mean_fnr", "under_covered", "median_area", "boundary_fnr", "interior_fnr"]}
    gout = {}
    for r in range(repeats or REPEATS):
        ci, ti = splits(S_cal, r)
        if not same:
            ti = np.arange(len(S_test.n))
        k = fit(S_cal, ci, method, alpha)
        fnr, area, bnd = apply(S_test, ti, method, k)
        out["mean_fnr"].append(fnr.mean()); out["under_covered"].append((fnr > BETA).mean())
        out["median_area"].append(np.nanmedian(area) if method != "argmax" else np.nan)
        out["boundary_fnr"].append(bnd.mean() if method != "argmax" else np.nan)
        nint = S_test.n[ti] - S_test.n_bnd[ti]
        if method != "argmax":
            out["interior_fnr"].append(((fnr * S_test.n[ti] - bnd * S_test.n_bnd[ti]) / np.maximum(nint, 1)).mean())
        for gname, gfun in (groups or {}).items():
            g = gfun(S_test, ti)
            for gv in np.unique(g):
                gout.setdefault(f"{gname}={gv}", {"mean_fnr": [], "under_covered": []})
                gout[f"{gname}={gv}"]["mean_fnr"].append(fnr[g == gv].mean())
                gout[f"{gname}={gv}"]["under_covered"].append((fnr[g == gv] > BETA).mean())
    res = {k: agg(v) for k, v in out.items() if v and not np.all(np.isnan(v))}
    res["groups"] = {g: {k: agg(v) for k, v in d.items()} for g, d in gout.items()}
    return res


def main():
    os.makedirs(FIG, exist_ok=True)
    R = {"alpha": ALPHA, "beta": BETA, "repeats": REPEATS, "models": {}}
    for m in MODELS:
        if not os.path.exists(os.path.join(CACHE, f"stats_{m}_voc_clean.npz")):
            continue
        S = Stats(os.path.join(CACHE, f"stats_{m}_voc_clean.npz"))
        q = np.quantile(S.n, [1 / 3, 2 / 3])
        groups = {"size": lambda s, i: np.digitize(s.n[i], q), "class": lambda s, i: s.cls[i]}
        M = R["models"][m] = {"miou": S.miou(), "n_objects": int(len(S.n)), "n_images": int(len(np.unique(S.img))),
                              "size_edges_px": q.tolist(), "clean": {}, "shift": {}, "alpha_sweep": {}}
        for meth in METHODS:
            M["clean"][meth] = evaluate(S, S, meth, groups)
        for meth in ["crc", "per-object"]:
            M["alpha_sweep"][meth] = {}
            for a in [0.02, 0.05, 0.1, 0.2, 0.3]:
                e = evaluate(S, S, meth, alpha=a, repeats=50)
                M["alpha_sweep"][meth][str(a)] = {k: e[k] for k in ["mean_fnr", "under_covered", "median_area"]}
        # distribution shift: calibrated on clean VOC, deployed on corrupted VOC / Oxford-IIIT Pet
        for f in sorted(glob.glob(os.path.join(CACHE, f"stats_{m}_*.npz"))):
            name = os.path.basename(f)[len(f"stats_{m}_"):-4]
            if name == "voc_clean":
                continue
            T = Stats(f)
            same = name.startswith("voc")
            M["shift"][name] = {"miou": T.miou() if same else None}
            for meth in ["argmax", "crc", "per-object"]:
                M["shift"][name][meth] = evaluate(S, T, meth, same=same)
                if meth != "argmax":   # recalibrated on data from the shifted condition itself
                    M["shift"][name][meth + "-recal"] = evaluate(T, T, meth)
        print(m, "mIoU %.3f" % M["miou"], {k: (round(v["mean_fnr"]["mean"], 3), round(v["under_covered"]["mean"], 3))
                                            for k, v in M["clean"].items()}, flush=True)
    json.dump(R, open(os.path.join(RES, "results.json"), "w"), indent=1)
    figures(R)


def figures(R):
    col = {"argmax": "#999999", "crc": "#1f4e9c", "crc-class": "#6a8fd6", "per-object": "#c0392b"}
    ms = list(R["models"])
    # 1. what each method guarantees: mean miss rate vs share of objects missed > 10 %
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    w = 0.8 / len(METHODS)
    for k, meth in enumerate(METHODS):
        for ax, key in zip(axs[:2], ["mean_fnr", "under_covered"]):
            v = [R["models"][m]["clean"][meth][key] for m in ms]
            ax.bar(np.arange(len(ms)) + (k - 1.5) * w, [x["mean"] for x in v], w, color=col[meth], label=MLABEL[meth],
                   yerr=[[x["mean"] - x["lo"] for x in v], [x["hi"] - x["mean"] for x in v]], capsize=2)
    axs[0].axhline(ALPHA, color="k", ls="--", lw=1); axs[1].axhline(ALPHA, color="k", ls="--", lw=1)
    axs[0].set_title("Mean share of object pixels missed"); axs[1].set_title("Objects with > 10 % of pixels missed")
    for ax in axs[:2]:
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([MODELS[m] for m in ms], fontsize=9); ax.grid(axis="y", alpha=.3)
    axs[0].legend(fontsize=7)
    for m, ls in zip(ms, ["-", "--"]):
        for meth in ["crc", "per-object"]:
            sw = R["models"][m]["alpha_sweep"][meth]
            a = sorted(sw, key=float)
            axs[2].plot([sw[x]["median_area"]["mean"] for x in a], [sw[x]["under_covered"]["mean"] for x in a], "o" + ls,
                        color=col[meth], label=f"{MODELS[m]}, {meth}")
    axs[2].set_xlabel("median mask area / true object area"); axs[2].set_ylabel("objects with > 10 % missed")
    axs[2].set_title("The price of safety (alpha = 0.02 ... 0.3)"); axs[2].legend(fontsize=7); axs[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "guarantees.png"), dpi=130); plt.close()

    # 2. who is left unprotected: per class and per size, CRC
    m = ms[0]
    G = R["models"][m]["clean"]["crc"]["groups"]
    cls = sorted([int(g.split("=")[1]) for g in G if g.startswith("class=")],
                 key=lambda c: -G[f"class={c}"]["mean_fnr"]["mean"])
    fig, axs = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [4, 1]})
    for k, meth in enumerate(["argmax", "crc", "crc-class"]):
        G = R["models"][m]["clean"][meth]["groups"]
        axs[0].bar(np.arange(len(cls)) + (k - 1) * 0.27, [G[f"class={c}"]["mean_fnr"]["mean"] for c in cls], 0.27,
                   color=col[meth], label=MLABEL[meth])
        axs[1].bar(np.arange(3) + (k - 1) * 0.27, [G[f"size={s}"]["mean_fnr"]["mean"] for s in range(3)], 0.27,
                   color=col[meth])
    axs[0].axhline(ALPHA, color="k", ls="--", lw=1); axs[1].axhline(ALPHA, color="k", ls="--", lw=1)
    axs[0].set_xticks(range(len(cls))); axs[0].set_xticklabels([CLASSES[c] for c in cls], rotation=60, fontsize=8)
    axs[0].set_ylabel("mean share of object pixels missed"); axs[0].legend(fontsize=8)
    axs[0].set_title(f"Per class ({MODELS[m]})")
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["small", "medium", "large"]); axs[1].set_title("Per object size")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "subgroups.png"), dpi=130); plt.close()

    # 3. shift
    fig, axs = plt.subplots(1, len(ms), figsize=(7.5 * len(ms), 4.2), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, ms):
        Sh = R["models"][m]["shift"]
        names = [f"voc_{c}_s3" for c in corruptions.CORRUPTIONS] + ["pets_clean"]
        names = [n for n in names if n in Sh]
        clean = R["models"][m]["clean"]
        rows = [("clean VOC", clean)] + [(n.replace("voc_", "").replace("_s3", " (sev. 3)").replace("pets_clean",
                                                                                                    "Oxford Pets"), Sh[n])
                                         for n in names]
        for k, (meth, lab) in enumerate([("argmax", "argmax"), ("crc", "risk control, calibrated clean"),
                                         ("crc-recal", "risk control, recalibrated")]):
            v = [r[meth]["mean_fnr"]["mean"] if meth in r else np.nan for _, r in rows]
            ax.bar(np.arange(len(rows)) + (k - 1) * 0.27, v, 0.27,
                   color={"argmax": "#999999", "crc": "#1f4e9c", "crc-recal": "#2ca02c"}[meth], label=lab)
        ax.axhline(ALPHA, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=20, fontsize=8)
        ax.set_title(MODELS[m]); ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("mean share of object pixels missed"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "shift.png"), dpi=130); plt.close()


if __name__ == "__main__":
    main()
