"""Evaluation (Phase 7).

    python evaluate.py                                  # best.pt on the test split
    python evaluate.py --split val                      # sanity-check against training logs
    python evaluate.py --checkpoint checkpoints/unet/best.pt --model unet

Reports CSI/IoU, FAR, POD and Brier score against the naive baselines, pooled and
stratified by fire size.

**The operating threshold comes from the checkpoint**, where it was selected on
validation. Re-selecting it on test would be choosing a hyperparameter on the test set;
the test-optimal value is printed too, but labelled as an oracle number that is not the
headline result.

Metrics are aggregated **per fire, not per tile**. Adjacent tiles overlap 50% and share
terrain and weather, so treating tiles as independent samples would give confidence
intervals that are far too narrow.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader

from model.convlstm_unet import build_model
from model.unet import UNet
from pipeline.dataset import WildfireDataset, channel_names, read_split
from pipeline.download import ROOT, load_config
from pipeline.features import FUEL_N_CLASSES

THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
# Dilation ring, in cells at 100 m. Replaces persistence as the naive baseline: against a
# new-burn target persistence predicts exactly the cells excluded by construction and
# scores 0. Measured on validation, the ring peaks at 5 cells (~500 m).
BASELINE_RINGS = (1, 3, 5, 8)


def scores(tp: float, fp: float, fn: float) -> dict:
    return {"csi": tp / max(tp + fp + fn, 1.0), "far": fp / max(tp + fp, 1.0),
            "pod": tp / max(tp + fn, 1.0)}


def fire_sizes() -> pd.Series:
    """Total burned cells per fire — a direct size measure, not a detection proxy."""
    out = {}
    for p in (ROOT / "data/processed/labels").glob("*.zarr"):
        g = zarr.open_group(p, mode="r")
        out[p.stem] = int((np.asarray(g["burn_new"]) > 0).sum())
    return pd.Series(out, name="burned_px")


@torch.no_grad()
def predict_split(cfg: dict, model, split: str, device: str, workers: int, amp: bool):
    """Per-sample TP/FP/FN at every threshold, plus Brier, keyed by fire."""
    ds = WildfireDataset(cfg, split, augment=False)
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                    num_workers=workers, pin_memory=True)
    n_thr = len(THRESHOLDS)
    rows = []
    model.eval()
    for bi, b in enumerate(dl):
        x, fuel, y = b["input"].to(device), b["fuel"].to(device), b["target"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            prob = torch.sigmoid(model(x, fuel).squeeze(1).float())
        t = y > 0.5
        brier = ((prob - y) ** 2).mean(dim=(1, 2)).cpu().numpy()
        tp = np.zeros((len(y), n_thr)); fp = np.zeros((len(y), n_thr)); fn = np.zeros((len(y), n_thr))
        for i, thr in enumerate(THRESHOLDS):
            p = prob > thr
            tp[:, i] = (p & t).sum(dim=(1, 2)).cpu().numpy()
            fp[:, i] = (p & ~t).sum(dim=(1, 2)).cpu().numpy()
            fn[:, i] = (~p & t).sum(dim=(1, 2)).cpu().numpy()
        for k in range(len(y)):
            rows.append({"fire_id": b["meta"]["fire_id"][k], "brier": float(brier[k]),
                         "tp": tp[k], "fp": fp[k], "fn": fn[k]})
        if bi % 20 == 0:
            print(f"  {bi*len(y)}/{len(ds)}", flush=True)
    return rows, ds


def baseline_counts(cfg: dict, split: str) -> dict:
    """Dilation-ring and persistence baselines, read straight from the label zarrs."""
    patch = int(cfg["grid"]["patch_size"])
    idx = pd.read_parquet(ROOT / "data/processed/sample_index.parquet")
    idx = idx[idx["fire_id"].isin(set(read_split(cfg, split)))]
    out = {k: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for k in ("persistence",) + BASELINE_RINGS}
    for fid, grp in idx.groupby("fire_id"):
        g = zarr.open_group(ROOT / "data/processed/labels" / f"{fid}.zarr", mode="r")
        burn = np.asarray(g["burn_new"]) > 0
        cum = np.cumsum(burn, axis=0) > 0
        for _, r in grp.iterrows():
            t, sl = int(r.t_index), (slice(int(r.row0), int(r.row0) + patch),
                                     slice(int(r.col0), int(r.col0) + patch))
            cur, y = cum[t][sl], burn[t + 1][sl] & ~cum[t][sl]
            for key in out:
                pred = cur if key == "persistence" else \
                    binary_dilation(cur, iterations=key) & ~cur
                out[key]["tp"] += float((pred & y).sum())
                out[key]["fp"] += float((pred & ~y).sum())
                out[key]["fn"] += float((~pred & y).sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint against baselines.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--model", choices=["convlstm_unet", "unet"], default="convlstm_unet")
    p.add_argument("--checkpoint")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()

    cfg = load_config(a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = a.checkpoint or f"checkpoints/{a.model}/best.pt"
    st = torch.load(ROOT / ckpt_path, map_location=device, weights_only=False)
    n_ch = len(channel_names(cfg))
    if a.model == "unet":
        model = UNet(n_ch, int(cfg["model"]["t_steps"]), FUEL_N_CLASSES,
                     int(cfg["model"].get("fuel_embed_dim", 8)),
                     tuple(cfg["model"]["hidden_dims"])).to(device)
    else:
        model = build_model(cfg, FUEL_N_CLASSES, n_ch).to(device)
    model.load_state_dict(st["model"])
    val_thr = float(st.get("threshold", 0.5))
    print(f"{ckpt_path}  epoch {st.get('epoch')}  val CSI {st.get('best_csi', float('nan')):.4f}")
    print(f"operating threshold {val_thr} (selected on validation)\n")
    print(f"evaluating {a.split}...")

    rows, ds = predict_split(cfg, model, a.split, device, a.workers,
                             bool(cfg["train"].get("amp", True)))
    df = pd.DataFrame(rows)
    ti = THRESHOLDS.index(val_thr) if val_thr in THRESHOLDS else THRESHOLDS.index(0.9)

    tp = np.stack(df["tp"].values); fp = np.stack(df["fp"].values); fn = np.stack(df["fn"].values)
    pooled = [scores(tp[:, i].sum(), fp[:, i].sum(), fn[:, i].sum()) for i in range(len(THRESHOLDS))]
    head = pooled[ti]
    best_i = int(np.argmax([s["csi"] for s in pooled]))

    print(f"\n=== {a.split}: {len(df):,} samples over {df.fire_id.nunique()} fires ===\n")
    print(f"{'':<34}{'CSI/IoU':>9}{'FAR':>8}{'POD':>8}")
    print("-" * 59)
    print(f"{'ConvLSTM U-Net @ val threshold':<34}{head['csi']:>9.4f}{head['far']:>8.3f}{head['pod']:>8.3f}")
    print(f"{'  (oracle: best test threshold '+str(THRESHOLDS[best_i])+')':<34}"
          f"{pooled[best_i]['csi']:>9.4f}{pooled[best_i]['far']:>8.3f}{pooled[best_i]['pod']:>8.3f}")
    print(f"{'  Brier score':<34}{df.brier.mean():>9.5f}")

    print("\nbaselines:")
    base = baseline_counts(cfg, a.split)
    for k, c in base.items():
        s = scores(c["tp"], c["fp"], c["fn"])
        nm = "persistence (= current burn)" if k == "persistence" else f"dilate {k} ring"
        print(f"  {nm:<32}{s['csi']:>9.4f}{s['far']:>8.3f}{s['pod']:>8.3f}")
    best_base = max((scores(c["tp"], c["fp"], c["fn"])["csi"] for c in base.values()))
    print(f"\n  model / best naive baseline: {head['csi']/max(best_base,1e-9):.2f}x")

    # Per fire, because 50%-overlapping tiles are not independent samples.
    per = df.groupby("fire_id").apply(
        lambda g: scores(np.stack(g["tp"].values)[:, ti].sum(),
                         np.stack(g["fp"].values)[:, ti].sum(),
                         np.stack(g["fn"].values)[:, ti].sum())["csi"], include_groups=False)
    print(f"\nper-fire CSI over {len(per)} fires: median {per.median():.4f}, "
          f"IQR {per.quantile(.25):.4f}-{per.quantile(.75):.4f}, mean {per.mean():.4f}")

    sizes = fire_sizes().reindex(per.index)
    q = pd.qcut(sizes, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"], duplicates="drop")
    print("\nby fire size (burned cells):")
    print(f"  {'quartile':<14}{'fires':>6}{'median CSI':>12}{'mean CSI':>10}")
    for lab, grp in per.groupby(q, observed=True):
        print(f"  {str(lab):<14}{len(grp):>6}{grp.median():>12.4f}{grp.mean():>10.4f}")

    out = ROOT / "runs" / f"{a.model}_{a.split}_eval.json"
    out.write_text(json.dumps({
        "checkpoint": str(ckpt_path), "split": a.split, "val_threshold": val_thr,
        "headline": head, "oracle": {"threshold": THRESHOLDS[best_i], **pooled[best_i]},
        "brier": float(df.brier.mean()),
        "baselines": {k: scores(c["tp"], c["fp"], c["fn"]) for k, c in base.items()},
        "per_fire_csi": {"median": float(per.median()), "mean": float(per.mean())},
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
