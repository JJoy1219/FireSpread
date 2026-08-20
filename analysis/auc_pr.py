"""Average precision (AUC-PR) on the test split, under two target definitions.

Published benchmarks report AUC-PR, so this exists to put a number next to theirs.
The comparison only means something if the target definitions are stated, because
they differ in a way that dominates the result.

  new      the target this project trains on: cells that burn in the next window,
           with already-burned cells EXCLUDED. Strictly harder, and very sparse.

  cumulative  the target Huot et al. and similar use: the fire mask at T+1,
           INCLUDING every already-burning cell. Those cells are trivially correct
           (they are given in the input), so they inflate both the base rate and
           the score. Emulated here by assigning probability 1 to already-burned
           cells and scoring against `burned OR new`.

AUC-PR must always be read against the positive base rate, which is the score a
random classifier gets. Lift over that rate is the comparable quantity, not the
raw number.

Average precision is computed from a probability histogram rather than by sorting
40 M pixels: exact to the bin width, and far cheaper in memory.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.dataset import WildfireDataset
from pipeline.download import load_config
from pipeline.viz_predict import load_model

BINS = 20000


class PRHist:
    """Positive/negative counts per probability bin, for average precision."""

    def __init__(self, bins: int = BINS):
        self.bins = bins
        self.pos = np.zeros(bins + 1, dtype="int64")
        self.neg = np.zeros(bins + 1, dtype="int64")

    def update(self, p: np.ndarray, y: np.ndarray) -> None:
        idx = np.clip((p * self.bins).astype("int32"), 0, self.bins)
        t = y > 0.5
        self.pos += np.bincount(idx[t], minlength=self.bins + 1)
        self.neg += np.bincount(idx[~t], minlength=self.bins + 1)

    def average_precision(self) -> float:
        """AP as the recall-weighted mean of precision, sweeping the threshold down."""
        P = self.pos.sum()
        if P == 0:
            return float("nan")
        tp = np.cumsum(self.pos[::-1])[::-1].astype("float64")
        fp = np.cumsum(self.neg[::-1])[::-1].astype("float64")
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
        rec = tp / P
        # Sum precision * (change in recall), from the highest-score bin downward.
        d_rec = np.diff(np.concatenate([[0.0], rec[::-1]]))
        return float((prec[::-1] * d_rec).sum())

    def base_rate(self) -> float:
        tot = self.pos.sum() + self.neg.sum()
        return float(self.pos.sum() / tot) if tot else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/unet_wide/best.pt")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--model", default="unet", choices=["unet", "convlstm_unet"])
    ap.add_argument("--hidden-dims", default="112,224,448")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["model"]["hidden_dims"] = [int(x) for x in a.hidden_dims.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = load_model(cfg, a.model, a.checkpoint, dev)
    model.eval()

    ds = WildfireDataset(cfg, a.split)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    burn_i = ds.channels.index("burn_mask")
    print(f"{a.checkpoint}  split {a.split}  {len(ds):,} samples  device {dev}")

    h_new, h_cum = PRHist(), PRHist()
    per_fire: dict[str, PRHist] = {}

    with torch.no_grad():
        for i, b in enumerate(dl):
            x, fuel, y = b["input"].to(dev), b["fuel"].to(dev), b["target"].to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x, fuel).squeeze(1)
            p = torch.sigmoid(logits.float()).cpu().numpy()
            yy = y.cpu().numpy()
            # Most recent step holds the fire's extent at prediction time.
            burned = (x[:, -1, burn_i] > 0.5).cpu().numpy()

            h_new.update(p.ravel(), yy.ravel())

            # Huot-style: already-burned cells are part of the target AND are given in
            # the input, so they score as certain hits.
            p_cum = np.where(burned, 1.0, p)
            y_cum = np.logical_or(burned, yy > 0.5).astype("float32")
            h_cum.update(p_cum.ravel(), y_cum.ravel())

            for j, fid in enumerate(b["meta"]["fire_id"]):
                per_fire.setdefault(str(fid), PRHist()).update(p[j].ravel(), yy[j].ravel())
            if i % 10 == 0:
                print(f"  {i * a.batch_size:,}/{len(ds):,}", flush=True)

    aps = np.array([h.average_precision() for h in per_fire.values()])
    aps = aps[~np.isnan(aps)]

    def row(name, h):
        apv, br = h.average_precision(), h.base_rate()
        print(f"  {name:34} AP {apv:.4f}   base rate {br:.4%}   lift {apv / br:6.1f}x")

    print("\nAverage precision (AUC-PR), pooled over all test pixels")
    row("new burn only (this project)", h_new)
    row("cumulative mask (Huot-style)", h_cum)
    print(f"\n  per fire, median AP {np.median(aps):.4f}   mean {aps.mean():.4f}   "
          f"n={len(aps)} fires")


if __name__ == "__main__":
    main()
