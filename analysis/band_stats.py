"""Positive base rate per distance band, on the train split only.

These rates set the per-band class weights in the loss, so they must come from
train alone -- deriving them from val or test would leak held-out statistics into
training, the same reason `norm_stats.json` is train-only.

Reads the label zarrs directly instead of building full samples. The expensive
part of a sample is regridding weather, which base rates do not need, so this
covers every training sample rather than a subsample.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import zarr

from pipeline.bands import BAND_LABELS, N_BANDS, band_index, distance_to_burn
from pipeline.download import ROOT, load_config
from pipeline.dataset import read_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--split", default="train", help="must be train for loss weights")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    a = ap.parse_args()

    cfg = load_config(a.config)
    patch = int(cfg["grid"]["patch_size"])
    t_steps = int(cfg["model"]["t_steps"])

    idx = pd.read_parquet(ROOT / cfg["paths"].get(
        "sample_index", "data/processed/sample_index.parquet"))
    fires = set(read_split(cfg, a.split))
    idx = idx[idx["fire_id"].isin(fires)].reset_index(drop=True)
    if a.limit:
        idx = idx.iloc[:a.limit]
    print(f"{len(idx):,} {a.split} samples over {idx['fire_id'].nunique()} fires", flush=True)

    npx = np.zeros(N_BANDS, dtype="int64")
    pos = np.zeros(N_BANDS, dtype="int64")
    groups: dict[str, zarr.Group] = {}
    skipped = 0

    for n, r in enumerate(idx.itertuples(index=False)):
        fid, t = str(r.fire_id), int(r.t_index)
        row0, col0 = int(r.row0), int(r.col0)
        if fid not in groups:
            groups[fid] = zarr.open_group(
                ROOT / "data/processed/labels" / f"{fid}.zarr", mode="r")
        burn = groups[fid]["burn_new"]
        sl = (slice(row0, row0 + patch), slice(col0, col0 + patch))

        cum = np.zeros((patch, patch), dtype=bool)
        for k in range(0, t + 1):
            cum |= np.asarray(burn[k][sl]) > 0
        target = (np.asarray(burn[t + 1][sl]) > 0) & ~cum
        if not cum.any():
            skipped += 1
            continue

        b = band_index(distance_to_burn(cum))
        # Already-burned pixels can never be a target, so exclude them from the
        # denominator. Leaving them in would deflate band 0's rate to zero and
        # make its weight meaningless.
        keep = ~cum
        np.add.at(npx, b[keep], 1)
        np.add.at(pos, b[keep & target], 1)
        if n % 500 == 0:
            print(f"  {n:,}/{len(idx):,}", flush=True)

    rate = np.where(npx > 0, pos / np.maximum(npx, 1), 0.0)
    out = {
        "split": a.split,
        "config": a.config,
        "n_samples": int(len(idx) - skipped),
        "edges_cells": [float(e) for e in __import__("pipeline.bands", fromlist=["EDGES"]).EDGES],
        "base_rate": [float(x) for x in rate],
        "n_pixels": [int(x) for x in npx],
        "n_positive": [int(x) for x in pos],
    }
    p = ROOT / (a.out or cfg["paths"].get("band_stats", "configs/band_stats.json"))
    p.write_text(json.dumps(out, indent=2))

    print(f"\n{'band':>12} {'base rate':>10} {'(1-p)/p':>10} {'pixels':>14} {'positives':>11}")
    for b in range(N_BANDS):
        w = (1 - rate[b]) / rate[b] if rate[b] > 0 else float("inf")
        print(f"{BAND_LABELS[b]:>12} {rate[b]:9.4%} {w:10.1f} {npx[b]:14,} {pos[b]:11,}")
    print(f"\nskipped {skipped} samples with no fire in tile")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
