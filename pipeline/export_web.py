"""Export whole-fire predictions as a self-contained bundle for the web viewer.

    python -m pipeline.export_web --fires 2021_3212,2023_2709,2022_3088,2022_3298

For each fire and each 24 h window this stitches the model's per-tile predictions back
into one fire-sized raster, alongside the burn-so-far mask, the actual new burn, and the
static layers. Everything is written as base64 PNGs inside a single JSON so the viewer can
be a self-contained page with no network access.

Two things the stitching has to get right:

* **Overlapping tiles.** Tiles sit on a 128 px stride, so most pixels are covered two to
  four times. Predictions are averaged over the covering tiles rather than overwritten,
  which would leave visible seams wherever the last tile happened to end.
* **Uncovered ground.** Tiles only cover the active perimeter, so most of the raster is
  never predicted. Those pixels are marked transparent rather than drawn as probability
  zero — "not asked" and "asked, answered no" are different claims.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from PIL import Image

from model.unet import UNet
from pipeline.dataset import WildfireDataset, channel_names
from pipeline.download import ROOT, load_config
from pipeline.features import FUEL_N_CLASSES, static_tile

MAX_PX = 448          # longest raster side in the bundle; keeps the page under the size cap
MAX_WINDOWS = 26      # per fire


def png_b64(arr: np.ndarray, alpha: np.ndarray | None = None, vmin=None, vmax=None) -> str:
    """Encode a 2-D float array as a base64 PNG, optionally with an alpha channel."""
    a = arr.astype("float32")
    lo = np.nanmin(a) if vmin is None else vmin
    hi = np.nanmax(a) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-6
    v = np.clip((a - lo) / (hi - lo), 0, 1)
    v = np.nan_to_num(v)
    g = (v * 255).astype("uint8")
    if alpha is None:
        img = Image.fromarray(g, mode="L")
    else:
        img = Image.fromarray(np.dstack([g, (alpha * 255).astype("uint8")]), mode="LA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _csi(pred: np.ndarray, truth: np.ndarray) -> float:
    """CSI over the whole fire raster for one window.

    Computed on the full-resolution arrays before downsampling — measuring it on the
    web-sized rasters would report the accuracy of the picture rather than the model.
    """
    tp = float((pred & truth).sum())
    return tp / max(tp + float((pred & ~truth).sum()) + float((~pred & truth).sum()), 1.0)


def downsample(a: np.ndarray, factor: int, how: str = "mean") -> np.ndarray:
    if factor <= 1:
        return a
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
    b = a[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return b.max(axis=(1, 3)) if how == "max" else b.mean(axis=(1, 3))


@torch.no_grad()
def stitch_window(model, ds, rows, H, W, patch, device):
    """Average per-tile predictions back onto the fire raster. Returns (prob, covered)."""
    acc = np.zeros((H, W), dtype="float32")
    cnt = np.zeros((H, W), dtype="float32")
    for start in range(0, len(rows), 8):
        chunk = rows[start:start + 8]
        xs, fs = [], []
        for i in chunk:
            s = ds[int(i)]
            xs.append(s["input"]); fs.append(s["fuel"])
        x = torch.stack(xs).to(device); f = torch.stack(fs).to(device)
        p = torch.sigmoid(model(x, f).squeeze(1).float()).cpu().numpy()
        for k, i in enumerate(chunk):
            r0 = int(ds.index.iloc[i].row0); c0 = int(ds.index.iloc[i].col0)
            acc[r0:r0 + patch, c0:c0 + patch] += p[k]
            cnt[r0:r0 + patch, c0:c0 + patch] += 1
    covered = cnt > 0
    out = np.zeros_like(acc)
    out[covered] = acc[covered] / cnt[covered]
    return out, covered


def wind_field(fire_id: str, cfg: dict, when: pd.Timestamp, n: int = 12):
    """Coarse u/v grid over the fire, for arrow overlays. Native HRRR cells, no regrid."""
    g = zarr.open_group(ROOT / cfg["paths"]["hrrr_windows"] / f"{fire_id}.zarr", mode="r")
    stamps = list(g.attrs["times"])
    key = f"{when:%Y%m%d_%H}z"
    if key not in stamps:
        return None
    i = stamps.index(key)
    if not bool(np.asarray(g["filled"])[i]):
        return None
    d = np.asarray(g["data"][i])
    u, v = d[0], d[1]
    ys = np.linspace(0, u.shape[0] - 1, n).astype(int)
    xs = np.linspace(0, u.shape[1] - 1, n).astype(int)
    return {"u": [[round(float(u[y, x]), 2) for x in xs] for y in ys],
            "v": [[round(float(v[y, x]), 2) for x in xs] for y in ys],
            "t2m": round(float(d[2].mean() - 273.15), 1),
            "rh": round(float(d[3].mean()), 1)}


def export_fire(fire_id: str, cfg: dict, model, ds, device, thr: float) -> dict | None:
    sub = ds.index[ds.index.fire_id == fire_id]
    if sub.empty:
        print(f"  {fire_id}: no samples in this split, skipping")
        return None
    patch = int(cfg["grid"]["patch_size"])
    lg = zarr.open_group(ROOT / "data/processed/labels" / f"{fire_id}.zarr", mode="r")
    burn = np.asarray(lg["burn_new"]) > 0
    cum = np.cumsum(burn, axis=0) > 0
    _, H, W = burn.shape
    times = [pd.Timestamp(t) for t in lg.attrs["window_times"]]
    factor = max(1, int(np.ceil(max(H, W) / MAX_PX)))

    windows = sorted(sub.t_index.unique())[:MAX_WINDOWS]
    frames = []
    for t in windows:
        rows = sub.index[sub.t_index == t].tolist()
        prob, covered = stitch_window(model, ds, rows, H, W, patch, device)
        truth = burn[t + 1] & ~cum[t]
        cur = cum[t]

        pd_s = downsample(prob, factor)
        cov_s = downsample(covered.astype("float32"), factor) > 0.2
        frames.append({
            "t": int(t),
            "date": times[t].strftime("%Y-%m-%d"),
            "pred": png_b64(pd_s, alpha=cov_s.astype("float32"), vmin=0, vmax=1),
            "truth": png_b64(downsample(truth.astype("float32"), factor, "max"),
                             alpha=downsample(truth.astype("float32"), factor, "max"),
                             vmin=0, vmax=1),
            "burned": png_b64(downsample(cur.astype("float32"), factor, "max"),
                              alpha=downsample(cur.astype("float32"), factor, "max"),
                              vmin=0, vmax=1),
            "new_px": int(truth.sum()),
            "burned_px": int(cur.sum()),
            "pred_px": int(((prob > thr) & covered).sum()),
            "csi": round(_csi((prob > thr) & covered, truth), 4),
            "wind": wind_field(fire_id, cfg, times[t]),
        })
        print(f"    window {t} {times[t]:%Y-%m-%d}  {len(rows)} tiles  "
              f"new {int(truth.sum())} px", flush=True)

    # Static layers, cropped to the same grid.
    sg = zarr.open_group(ROOT / "data/processed/features" / f"{fire_id}.zarr", mode="r")
    chans = list(sg.attrs["channels"])
    stat = np.asarray(sg["static"]).astype("float32")
    layers = {}
    for name, key in (("elevation", "elevation"), ("slope", "slope"),
                      ("canopy_cover", "cc"), ("canopy_height", "ch")):
        a = downsample(stat[chans.index(key)], factor)
        layers[name] = {"png": png_b64(a),
                        "min": round(float(np.nanmin(a)), 1),
                        "max": round(float(np.nanmax(a)), 1)}
    return {
        "fire_id": fire_id, "split": ds.split,
        "height": int(np.ceil(H / factor)), "width": int(np.ceil(W / factor)),
        # Two different resolutions, and conflating them is a 16x area error on the big
        # fires: `res_m` is the DISPLAY pixel size after downsampling, while every pixel
        # COUNT below is measured on the native grid. Areas must use `native_res_m`.
        "res_m": int(cfg["grid"]["resolution_m"] * factor),
        "native_res_m": int(cfg["grid"]["resolution_m"]),
        "threshold": thr, "frames": frames, "static": layers,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Export fire predictions for the web viewer.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--checkpoint", default="checkpoints/unet_wide/best.pt")
    p.add_argument("--hidden-dims", default="112,224,448")
    p.add_argument("--fires", default="2021_3212,2023_2709,2022_3088,2022_3298")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/processed/web_bundle.json")
    a = p.parse_args()

    cfg = load_config(a.config)
    cfg["model"]["hidden_dims"] = [int(v) for v in a.hidden_dims.split(",")]
    st = torch.load(ROOT / a.checkpoint, map_location=a.device, weights_only=False)
    n_ch = len(channel_names(cfg))
    model = UNet(n_ch, int(cfg["model"]["t_steps"]), FUEL_N_CLASSES,
                 int(cfg["model"].get("fuel_embed_dim", 8)),
                 tuple(cfg["model"]["hidden_dims"])).to(a.device)
    model.load_state_dict(st["model"]); model.eval()
    thr = float(st.get("threshold", 0.5))

    want = [f.strip() for f in a.fires.split(",") if f.strip()]
    ds_cache = {sp: WildfireDataset(cfg, sp, augment=False) for sp in ("val", "test")}
    out = {"threshold": thr, "checkpoint": a.checkpoint, "fires": []}
    for fid in want:
        ds = next((d for d in ds_cache.values()
                   if (d.index.fire_id == fid).any()), None)
        if ds is None:
            print(f"  {fid}: not in val or test, skipping")
            continue
        print(f"{fid} ({ds.split})")
        rec = export_fire(fid, cfg, model, ds, a.device, thr)
        if rec:
            out["fires"].append(rec)

    dest = ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    mb = dest.stat().st_size / 1e6
    print(f"\n{len(out['fires'])} fires, {sum(len(f['frames']) for f in out['fires'])} frames"
          f"  ->  {dest}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
