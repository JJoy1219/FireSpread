"""Burn mask generation from FIRMS detections (Phase 2).

For each fire and each window (24 h — see `labels.window_hours` and the README on why
6 h does not work with a single polar orbiter), rasterise that window's detections onto
the 100 m EPSG:5070 modelling grid, dilate to the VIIRS footprint, then close gaps.

What is stored is **new detections per window**, not cumulative masks. The cumulative
label at T is the running OR up to T and the target is the OR over the horizon, both
cheap to derive at load time. Storing the increments keeps one representation that
serves any `t_horizon_h` and avoids writing the same burned pixels into every later
timestep — see the storage budget in the README.

    python -m pipeline.labels --all-mvp
    python -m pipeline.labels --fire-id 2018_4037
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.features
import zarr
from affine import Affine
from scipy.ndimage import binary_closing, binary_dilation

from pipeline.download import ROOT, label_dir, load_config


def fire_grid(row: pd.Series, margin_m: float, res: float) -> tuple[Affine, int, int]:
    """Grid covering the fire plus margin, snapped so cells align across all fires."""
    x0 = np.floor((row["x_min"] - margin_m) / res) * res
    y0 = np.floor((row["y_min"] - margin_m) / res) * res
    x1 = np.ceil((row["x_max"] + margin_m) / res) * res
    y1 = np.ceil((row["y_max"] + margin_m) / res) * res
    width = int((x1 - x0) / res)
    height = int((y1 - y0) / res)
    # North-up transform: origin is the top-left corner.
    return Affine(res, 0, x0, 0, -res, y1), width, height


def _disk(radius: int) -> np.ndarray:
    r = int(radius)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def rasterize_window(x: np.ndarray, y: np.ndarray, transform: Affine,
                     shape: tuple[int, int], dilation_px: int,
                     closing_px: int = 0) -> np.ndarray:
    """Points -> binary raster: footprint dilation, then gap-closing.

    Two separate jobs, which CLAUDE.md conflates into one dilation:

    1. `dilation_px` restores the VIIRS footprint. A detection is a 375 m pixel, not a
       point, and rasterising it onto a 100 m grid marks a single cell.
    2. `closing_px` bridges gaps *between* detections. Fronts are not detected at every
       pixel and scan geometry widens spacing toward the swath edge, so dilation alone
       leaves a stippled, disconnected perimeter. Closing (dilate then erode) joins them
       without pushing the outer boundary outward the way more dilation would.

    (The most faithful footprint would use each detection's own `scan`/`track` extent,
    which FIRMS provides but `load_detections` currently drops.)
    """
    if len(x) == 0:
        return np.zeros(shape, dtype=np.uint8)
    shapes = ({"type": "Point", "coordinates": (float(a), float(b))} for a, b in zip(x, y))
    arr = rasterio.features.rasterize(
        ((s, 1) for s in shapes), out_shape=shape, transform=transform,
        fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    if dilation_px > 0:
        arr = binary_dilation(arr, structure=_disk(dilation_px))
    if closing_px > 0:
        arr = binary_closing(arr, structure=_disk(closing_px))
    return arr.astype(np.uint8)


def build_fire(row: pd.Series, det: pd.DataFrame, cfg: dict, out_dir: Path,
               overwrite: bool = False) -> dict:
    lc, gc = cfg["labels"], cfg["grid"]
    res = float(gc["resolution_m"])
    window_h = int(lc["window_hours"])
    margin_m = gc["patch_size"] * res / 2 + 2000   # half a patch, plus slack for dilation

    dest = out_dir / f"{row['fire_id']}.zarr"
    if dest.exists():
        if not overwrite:
            return {"fire_id": row["fire_id"], "status": "cached"}
        shutil.rmtree(dest)

    d = det[det["fire_id"] == row["fire_id"]].copy()
    d["bin"] = pd.to_datetime(d["acq_datetime"]).dt.floor(f"{window_h}h")

    transform, width, height = fire_grid(row, margin_m, res)
    bins = pd.date_range(d["bin"].min(), d["bin"].max(), freq=f"{window_h}h")
    groups = {b: g for b, g in d.groupby("bin")}

    # Gap rule: a window is unusable if too long has passed since the last window that
    # actually had detections — absence there means "no observation", not "no fire".
    has_det = np.array([b in groups for b in bins], dtype=bool)
    gap_h = np.zeros(len(bins))
    last = None
    for i, b in enumerate(bins):
        gap_h[i] = 0 if last is None else (b - last).total_seconds() / 3600
        if has_det[i]:
            last = b
    gap_flag = gap_h > lc["max_gap_hours"]

    # No-data windows from the sensor-outage list (config firms.no_data_windows).
    nodata = np.zeros(len(bins), dtype=bool)
    for w0, w1 in (cfg["firms"].get("no_data_windows") or []):
        a = pd.Timestamp(w0, tz="UTC")
        z = pd.Timestamp(w1, tz="UTC") + pd.Timedelta(days=1)
        nodata |= np.array([(a <= b < z) for b in bins])

    grp = zarr.open_group(dest, mode="w")
    arr = grp.create_array("burn_new", shape=(len(bins), height, width), dtype="uint8",
                           chunks=(1, min(512, height), min(512, width)))
    n_px = 0
    for i, b in enumerate(bins):
        g = groups.get(b)
        m = (rasterize_window(g["x"].to_numpy(), g["y"].to_numpy(), transform,
                              (height, width), lc["dilation_px"], lc.get("closing_px", 0))
             if g is not None else np.zeros((height, width), dtype=np.uint8))
        arr[i] = m
        n_px += int(m.sum())

    grp.attrs.update({
        "fire_id": row["fire_id"],
        "crs": cfg["region"]["crs"],
        "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        "resolution_m": res,
        "window_hours": window_h,
        "window_times": [b.isoformat() for b in bins],
        "has_detections": has_det.tolist(),
        "gap_flag": gap_flag.tolist(),
        "no_data": nodata.tolist(),
        # `usable` means "this window may serve as a prediction TARGET". An unobserved
        # window is not evidence the fire stopped, so a sample is only valid when the
        # target window is usable. Cumulative labels stay valid through empty windows.
        "usable": (has_det & ~gap_flag & ~nodata).tolist(),
        "dilation_px": lc["dilation_px"],
        "closing_px": lc.get("closing_px", 0),
    })
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    return {
        "fire_id": row["fire_id"], "status": "built", "windows": len(bins),
        "usable": int((has_det & ~gap_flag & ~nodata).sum()), "gaps": int(gap_flag.sum()),
        "shape": f"{height}x{width}", "burned_px": n_px, "mb": round(size_mb, 2),
        "observed": int(has_det.sum()),
        "raw_mb": round(len(bins) * height * width / 1e6, 1),
    }


def main() -> None:
    from pipeline.hrrr import pick_mvp_fires

    p = argparse.ArgumentParser(description="Rasterise FIRMS detections into burn masks.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--events", default="data/processed/fire_events.csv")
    p.add_argument("--fire-id")
    p.add_argument("--all-mvp", action="store_true")
    p.add_argument("--all", action="store_true", help="every kept fire (692)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    det = pd.read_parquet(ROOT / "data/processed/detections_labeled.parquet")
    out_dir = label_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        ev = pd.read_csv(ROOT / args.events, parse_dates=["start", "end"])
        fires = ev[ev["keep"]]
    else:
        fires = pick_mvp_fires(ROOT / args.events)
        if args.fire_id:
            fires = fires[fires["fire_id"] == args.fire_id]
        elif not args.all_mvp:
            raise SystemExit("pass --fire-id, --all-mvp or --all")

    rows = []
    for i, (_, f) in enumerate(fires.iterrows(), 1):
        r = build_fire(f, det, cfg, out_dir, args.overwrite)
        rows.append(r)
        if r["status"] == "built":
            print(f"[{i}/{len(fires)}] {r['fire_id']}  {r['shape']}  "
                  f"{r['windows']} windows ({r['usable']} usable, {r['gaps']} gap-flagged)  "
                  f"{r['burned_px']:,} burned px  {r['mb']} MB "
                  f"({r['raw_mb']:.0f} MB raw, {r['raw_mb']/max(r['mb'],1e-9):.0f}x compressed)")

    man = pd.DataFrame(rows)
    out = out_dir.parent / f"{out_dir.name}_manifest.csv"
    man.to_csv(out, index=False)
    if "mb" in man:
        print(f"\n{len(man)} fires, {man['mb'].sum():.1f} MB on disk "
              f"(vs {man['raw_mb'].sum():.0f} MB uncompressed)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
