"""Baseline far-field CSI on VALIDATION, for a matched comparison with the per-band arm.

The per-band arm reports far CSI on validation. The only baseline far-field numbers
so far come from the test split, and validation is 2021 (Dixie, Caldor) against a
2022-23 test of mostly small fires, so quoting a ratio across them would be
meaningless. This measures the baseline checkpoint on the same split, same bands,
same thresholds.

Uses configs/perband.yaml purely to get the Dataset to emit the distance-band map.
The map is not an input to the model, and every other key is identical to baseline,
so the inputs the network sees are unchanged.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from pipeline.bands import BAND_LABELS, FAR_BAND_MIN, N_BANDS
from pipeline.dataset import WildfireDataset
from pipeline.download import ROOT, load_config
from pipeline.viz_predict import load_model
from torch.utils.data import DataLoader
from train import THRESHOLDS, BandCounts, Counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/unet_wide/best.pt")
    ap.add_argument("--model", default="unet", choices=["unet", "convlstm_unet"])
    ap.add_argument("--hidden-dims", default="112,224,448")
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", default="configs/perband.yaml",
                    help="only used so the Dataset emits the band map")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["model"]["hidden_dims"] = [int(x) for x in a.hidden_dims.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_thr, _ = load_model(cfg, a.model, a.checkpoint, dev)
    model.eval()

    ds = WildfireDataset(cfg, a.split)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    print(f"{a.checkpoint}  split {a.split}  {len(ds):,} samples  device {dev}")

    counts = Counts.zeros(len(THRESHOLDS))
    bands = {t: BandCounts() for t in THRESHOLDS}
    with torch.no_grad():
        for i, b in enumerate(dl):
            x, fuel = b["input"].to(dev), b["fuel"].to(dev)
            y, bd = b["target"].to(dev), b["band"].to(dev)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x, fuel).squeeze(1)
            p = torch.sigmoid(logits.float())
            counts.update(p, y)
            for t in THRESHOLDS:
                bands[t].update(p, y, bd, t)
            if i % 40 == 0:
                print(f"  {i*a.batch_size:,}/{len(ds):,}", flush=True)

    mets = counts.metrics()
    best = max(mets, key=lambda m: m["csi"])
    best_far_thr = max(THRESHOLDS, key=lambda t: bands[t].far_csi())

    print(f"\npooled CSI {best['csi']:.4f} @ {best['threshold']}")
    print(f"far CSI at the pooled-best threshold {best['threshold']}: "
          f"{bands[best['threshold']].far_csi():.4f}")
    print(f"far CSI at its OWN best threshold {best_far_thr}: "
          f"{bands[best_far_thr].far_csi():.4f}   <- the fair number to compare")
    print(f"far CSI at the checkpoint's stored threshold {ckpt_thr}: "
          f"{bands[ckpt_thr].far_csi():.4f}" if ckpt_thr in bands else "")

    bc = bands[best_far_thr].csi()
    print(f"\nCSI by distance @ {best_far_thr}:")
    for b in range(N_BANDS):
        if not np.isnan(bc[b]):
            print(f"  {BAND_LABELS[b]:>12}  {bc[b]:.4f}")
    print(f"\nfar = bands from {BAND_LABELS[FAR_BAND_MIN]} outward")


if __name__ == "__main__":
    main()
