"""Does wind matter more in daytime burn windows than overnight ones?

The sharpest test the 12 h dataset makes possible. At 24 h every window averages one
afternoon run together with one overnight lull, so any wind response is diluted. At
12 h the windows separate, and S-NPP's two daily overpasses mean each window sits
mostly on one side of the diurnal cycle.

Fire spread is strongly diurnal: afternoons are hot, dry and windy with active runs,
nights are cool, humid and calm. If the model uses wind as fire weather at all, the
effect should be concentrated in daytime windows. If wind is equally irrelevant in
both, that is strong evidence the signal is not being used rather than being masked
by averaging.

Occlusion is by permutation across the batch, matching viz_predict.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import zarr

from pipeline.dataset import WildfireDataset
from pipeline.download import ROOT, label_dir, load_config
from pipeline.viz_predict import CHANNEL_GROUPS, load_model


def window_hour(fid: str, t: int, cfg: dict, cache: dict) -> int | None:
    """UTC hour of the label window this sample predicts from."""
    if fid not in cache:
        g = zarr.open_group(label_dir(cfg) / f"{fid}.zarr", mode="r")
        cache[fid] = [pd.Timestamp(x) for x in g.attrs["window_times"]]
    times = cache[fid]
    return times[t].hour if 0 <= t < len(times) else None


@torch.no_grad()
def run(model, ds, idx, cfg, dev, thr, n_max):
    ch = {c: i for i, c in enumerate(ds.channels)}
    rng = np.random.default_rng(0)
    take = idx if len(idx) <= n_max else list(rng.choice(idx, size=n_max, replace=False))
    tp = {k: 0.0 for k in list(CHANNEL_GROUPS) + ["_base"]}
    fp = dict(tp); fn = dict(tp)
    dP = {k: [] for k in CHANNEL_GROUPS}

    xs, fs, ys = [], [], []
    for i in take:
        s = ds[int(i)]
        xs.append(s["input"]); fs.append(s["fuel"]); ys.append(s["target"])
    X = torch.stack(xs).to(dev); F = torch.stack(fs).to(dev); Y = torch.stack(ys).to(dev)

    def score(key, P):
        pred = P > thr
        t = Y > 0.5
        tp[key] += float((pred & t).sum()); fp[key] += float((pred & ~t).sum())
        fn[key] += float((~pred & t).sum())

    base = torch.sigmoid(model(X, F).squeeze(1))
    score("_base", base)
    # Region of interest: near the fire. Far pixels are ~99% of the tile and always
    # near zero, so including them dilutes every effect toward nothing.
    roi = (X[:, -1, ch["burn_mask"]] > 0.5) | (Y > 0.5)
    roi = roi | torch.nn.functional.max_pool2d(roi.float()[:, None], 21, 1, 10)[:, 0].bool()

    for name, cols in CHANNEL_GROUPS.items():
        Xp = X.clone()
        perm = torch.randperm(Xp.shape[0], device=dev)
        for c in cols:
            if c in ch:
                Xp[:, :, ch[c]] = X[perm][:, :, ch[c]]
        P = torch.sigmoid(model(Xp, F).squeeze(1))
        score(name, P)
        d = (P - base).abs()[roi]
        dP[name].append(float(d.mean()) if d.numel() else 0.0)

    def csi(k):
        den = tp[k] + fp[k] + fn[k]
        return tp[k] / den if den else 0.0
    return csi("_base"), {k: csi("_base") - csi(k) for k in CHANNEL_GROUPS}, \
           {k: float(np.mean(v)) for k, v in dP.items()}, len(take)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/h12.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/h12_s1/best.pt")
    ap.add_argument("--hidden-dims", default="112,224,448")
    ap.add_argument("--split", default="val")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--n", type=int, default=64, help="samples per group")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["model"]["hidden_dims"] = [int(x) for x in a.hidden_dims.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_thr, _ = load_model(cfg, "unet", a.checkpoint, dev)
    thr = a.threshold if a.threshold is not None else ckpt_thr
    ds = WildfireDataset(cfg, a.split)

    cache: dict = {}
    groups: dict[str, list[int]] = {"day": [], "night": []}
    for i in range(len(ds)):
        r = ds.index.iloc[i]
        h = window_hour(str(r["fire_id"]), int(r["t_index"]), cfg, cache)
        if h is None:
            continue
        # labels.py floors detections into fixed bins, so at window_hours=12 the only
        # window times are 00Z and 12Z. Bucket by which overpass each bin contains:
        #   12Z bin spans 12-24Z, holding the 19-22Z pass = 12:00-15:00 local, AFTERNOON
        #   00Z bin spans 00-12Z, holding the 08-11Z pass = 01:00-04:00 local, NIGHT
        # An earlier version tested the raw hour against an overpass-time range and put
        # every window in one bucket, since neither 0 nor 12 falls in 15-24.
        groups["day" if h == 12 else "night"].append(i)

    print(f"{a.checkpoint}  split {a.split}  threshold {thr}")
    print(f"day windows {len(groups['day']):,}   night windows {len(groups['night']):,}\n")

    for g, idx in groups.items():
        if not idx:
            print(f"{g}: no samples"); continue
        base, drops, dP, n = run(model, ds, idx, cfg, dev, thr, a.n)
        print(f"--- {g} ({n} sampled) --- baseline CSI {base:.4f}")
        for k in sorted(drops, key=lambda k: -drops[k]):
            print(f"    {k:34} dP {dP[k]:.4f}   CSI drop {drops[k]:+.4f}")
        print()


if __name__ == "__main__":
    main()
