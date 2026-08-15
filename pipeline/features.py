"""Feature stack construction and the tiled sample index (Phase 3).

Everything is resampled onto **the exact grid the labels already use** — the transform
and shape are read back from each fire's label zarr rather than recomputed, so features
and targets cannot drift apart by a half pixel.

Two halves:

* `build_static` — terrain and fuel. Static per fire, so computed once.
* `build_index`  — enumerates (fire_id, timestep, tile) samples over the active
  perimeter, which is the option-A tiling scheme (see README).

The weather half is deliberately not here yet: 24 h windows with `t_steps: 3` need HRRR
from T-48h, and the current MVP pull only covers T-12h to T+24h. See `hrrr_coverage`.

    python -m pipeline.features static --all-mvp
    python -m pipeline.features index --all-mvp
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import zarr
from affine import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

from pipeline.download import ROOT, load_config

# Scott & Burgan 40 codes are sparse (91-204). Map them to a dense index so the model
# can use a small embedding table rather than a 205-wide one-hot.
FUEL_GROUPS = {
    "non-burnable": (91, 99), "grass": (101, 109), "grass-shrub": (121, 129),
    "shrub": (141, 149), "timber-understory": (161, 169),
    "timber-litter": (181, 189), "slash": (201, 209),
}
CONT_CHANNELS = ["elevation", "slope", "aspect_sin", "aspect_cos", "tpi", "cc", "ch"]


def label_grid(fire_id: str) -> tuple[Affine, int, int, str]:
    """Read the authoritative grid back from the label zarr."""
    g = zarr.open_group(ROOT / "data/processed/labels" / f"{fire_id}.zarr", mode="r")
    t = g.attrs["transform"]
    h, w = g["burn_new"].shape[1:]
    return Affine(*t), h, w, g.attrs["crs"]


def warp_to_grid(src_path: Path, band: int, transform: Affine, shape: tuple[int, int],
                 crs: str, resampling: Resampling, dtype="float32") -> np.ndarray:
    """Reproject/resample one band of a raster onto the target grid."""
    dst = np.zeros(shape, dtype=dtype)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=resampling, src_nodata=src.nodata, dst_nodata=np.nan,
        )
    return dst


def terrain_derivatives(dem: np.ndarray, res: float, tpi_radius_m: float = 300.0):
    """Slope, aspect (as sin/cos) and TPI from a north-up DEM in projected metres.

    Note on resolution: these are computed on the 100 m modelling grid, so they measure
    the net gradient over a 200 m baseline. Deriving them at the DEM's native resolution
    and then aggregating would retain sub-grid roughness, which fire actually responds
    to — but the full dataset will use a statewide 100 m DEM, and consistency between
    the MVP fires and the full run matters more than the extra fidelity here.
    """
    # Row index increases southward on a north-up raster, hence the sign flip.
    dz_drow, dz_dcol = np.gradient(dem, res, res)
    dz_deast = dz_dcol
    dz_dnorth = -dz_drow

    slope_deg = np.degrees(np.arctan(np.hypot(dz_deast, dz_dnorth)))

    # Aspect = compass bearing the hillside FACES, i.e. the downhill direction.
    aspect = np.arctan2(-dz_deast, -dz_dnorth)
    aspect_sin = np.sin(aspect)
    aspect_cos = np.cos(aspect)
    # Aspect is undefined on flat ground; leave it at the origin rather than inventing one.
    flat = slope_deg < 0.1
    aspect_sin[flat] = 0.0
    aspect_cos[flat] = 0.0

    size = max(int(round(2 * tpi_radius_m / res)) | 1, 3)   # odd window
    tpi = dem - uniform_filter(dem, size=size, mode="nearest")

    return slope_deg, aspect_sin, aspect_cos, tpi


def fuel_dense_index(codes: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map sparse Scott-Burgan codes onto 0..N-1, with 0 reserved for nodata."""
    present = sorted(int(c) for c in np.unique(codes) if c > 0)
    mapping = {c: i + 1 for i, c in enumerate(present)}
    out = np.zeros(codes.shape, dtype=np.int16)
    for c, i in mapping.items():
        out[codes == c] = i
    return out, mapping


def build_static(fire_id: str, cfg: dict, overwrite: bool = False) -> dict:
    transform, h, w, crs = label_grid(fire_id)
    res = float(cfg["grid"]["resolution_m"])
    dest = ROOT / "data/processed/features" / f"{fire_id}.zarr"
    if dest.exists():
        if not overwrite:
            return {"fire_id": fire_id, "status": "cached"}
        shutil.rmtree(dest)

    dem_p = ROOT / "data/raw/dem" / f"{fire_id}_dem.tif"
    lf_p = sorted(glob.glob(str(ROOT / "data/raw/landfire" / f"{fire_id}_*.tif")))
    if not dem_p.exists() or not lf_p:
        return {"fire_id": fire_id, "status": "missing raw layers"}
    lf_p = Path(lf_p[0])

    # Elevation is continuous -> bilinear.
    dem = warp_to_grid(dem_p, 1, transform, (h, w), crs, Resampling.bilinear)
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem[np.isfinite(dem)]))
    slope, asp_sin, asp_cos, tpi = terrain_derivatives(dem, res)

    # LANDFIRE is already EPSG:5070 at 30 m, so this is a pure downsample. FBFM40 is
    # categorical and must use majority — averaging fuel-model codes would invent
    # classes that do not exist. CC/CH are continuous and take the area average.
    fuel_raw = warp_to_grid(lf_p, 1, transform, (h, w), crs, Resampling.mode)
    cc = warp_to_grid(lf_p, 2, transform, (h, w), crs, Resampling.average)
    ch = warp_to_grid(lf_p, 3, transform, (h, w), crs, Resampling.average)
    fuel_raw = np.where(np.isfinite(fuel_raw), fuel_raw, 0).astype(np.int16)
    cc = np.nan_to_num(cc)
    ch = np.nan_to_num(ch) / 10.0        # LANDFIRE CH ships in metres x 10
    fuel_idx, fuel_map = fuel_dense_index(fuel_raw)

    cont = np.stack([dem, slope, asp_sin, asp_cos, tpi, cc, ch]).astype(np.float16)

    grp = zarr.open_group(dest, mode="w")
    grp.create_array("static", shape=cont.shape, dtype="float16",
                     chunks=(cont.shape[0], min(512, h), min(512, w)))[:] = cont
    grp.create_array("fuel", shape=(h, w), dtype="int16",
                     chunks=(min(512, h), min(512, w)))[:] = fuel_idx
    grp.attrs.update({
        "fire_id": fire_id, "crs": crs, "resolution_m": res,
        "transform": [transform.a, transform.b, transform.c,
                      transform.d, transform.e, transform.f],
        "channels": CONT_CHANNELS,
        "fuel_code_to_index": {str(k): v for k, v in fuel_map.items()},
        "fuel_n_classes": len(fuel_map) + 1,
    })
    mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    return {
        "fire_id": fire_id, "status": "built", "shape": f"{h}x{w}",
        "elev_m": f"{dem.min():.0f}-{dem.max():.0f}",
        "slope_deg_p95": round(float(np.percentile(slope, 95)), 1),
        "fuel_classes": len(fuel_map), "cc_max": round(float(cc.max()), 1),
        "ch_max_m": round(float(ch.max()), 1), "mb": round(mb, 2),
    }


def hrrr_coverage(fire_id: str, cfg: dict) -> dict:
    """How much of the weather history the stored HRRR windows can actually serve.

    Counts the windowed zarr, not raw GRIBs — those are deleted after extraction
    (`storage.keep_raw_hrrr_grib`), so globbing them reports 0% forever. `hours_needed`
    is the exact feature-driven set from `hrrr.needed_hours`, which is what gets
    fetched; the old hourly count over the whole span overstated it ~8x and also
    started 12 h too late to cover the earliest wind lag.
    """
    from pipeline.hrrr import needed_hours

    hours = needed_hours(fire_id, cfg)
    store = ROOT / cfg["paths"]["hrrr_windows"] / f"{fire_id}.zarr"
    present = extra = 0
    if store.exists():
        g = zarr.open_group(store, mode="r")
        # The store is a superset of the needed set: gap repair adds +/-1 h bracketing
        # hours. Count membership rather than comparing the lists, or repair would look
        # like a coverage failure.
        have = {s for s, f in zip(g.attrs.get("times", []), np.asarray(g["filled"])) if f}
        present = sum(1 for t in hours if f"{t:%Y%m%d_%H}z" in have)
        extra = len(have) - present
    return {
        "fire_id": fire_id, "need_from": str(hours[0]), "need_to": str(hours[-1]),
        "hours_needed": len(hours), "hours_present": present, "repair_hours": extra,
        "pct": round(100 * present / max(len(hours), 1), 1),
    }


def tile_origins(lo: int, hi: int, n: int, patch: int, stride: int) -> list[int]:
    """Tile origins along one axis covering [lo, hi], clamped inside [0, n - patch].

    Deliberately *not* snapped to a global lattice shared across fires. Tiles never need
    to align between fires, and lattice points do not reliably fall inside a small fire's
    raster — which silently produced zero samples for compact fires.
    """
    if n <= patch:
        return [0]
    last = n - patch
    if hi - lo + 1 <= patch:                      # active area fits one tile: centre it
        return [max(0, min((lo + hi) // 2 - patch // 2, last))]
    outs = list(range(max(0, min(lo, last)), min(hi, last) + 1, stride))
    tail = max(0, min(hi - patch + 1, last))      # guarantee the far edge is covered
    if not outs or outs[-1] < tail:
        outs.append(tail)
    return sorted({max(0, min(o, last)) for o in outs})


def build_index(fire_id: str, cfg: dict) -> pd.DataFrame:
    """Enumerate (fire_id, timestep, tile) samples covering the active perimeter."""
    sc, gc = cfg["sampling"], cfg["grid"]
    patch = int(gc["patch_size"])
    stride = int(sc["tile_stride_px"])
    res = float(gc["resolution_m"])
    t_steps = int(cfg["model"]["t_steps"])

    g = zarr.open_group(ROOT / "data/processed/labels" / f"{fire_id}.zarr", mode="r")
    burn = g["burn_new"]
    usable = np.array(g.attrs["usable"], dtype=bool)
    times = [pd.Timestamp(t) for t in g.attrs["window_times"]]
    tr = Affine(*g.attrs["transform"])
    n_t, H, W = burn.shape

    cum = np.zeros((H, W), dtype=bool)
    cums = []
    for t in range(n_t):
        cum = cum | (burn[t] > 0)
        cums.append(cum.copy())

    rows = []
    # Need t_steps-1 windows of history behind t, and window t+1 as the target.
    for t in range(t_steps - 1, n_t - 1):
        if not usable[t] or not usable[t + 1]:
            continue
        active = burn[t] > 0                      # cells burning in this window
        if not active.any():
            continue
        target = (burn[t + 1] > 0) & ~cums[t]     # genuinely new burn
        if not target.any():
            continue

        ar, ac = np.nonzero(active)
        for row0 in tile_origins(int(ar.min()), int(ar.max()), H, patch, stride):
            for col0 in tile_origins(int(ac.min()), int(ac.max()), W, patch, stride):
                sl = (slice(row0, row0 + patch), slice(col0, col0 + patch))
                n_active = int(active[sl].sum())
                n_target = int(target[sl].sum())
                if n_active == 0 and n_target == 0:
                    continue
                # Phase 8 clipping: growth reaching this tile's edge almost certainly
                # continues past it, so the target is truncated and the sample teaches
                # the model that the fire stopped at an arbitrary boundary.
                tt = target[sl]
                clipped = bool(tt[0].any() or tt[-1].any() or tt[:, 0].any() or tt[:, -1].any())
                rows.append({
                    "fire_id": fire_id, "t_index": t, "window_time": times[t].isoformat(),
                    "row0": row0, "col0": col0,
                    "x0_m": tr.c + col0 * res, "y1_m": tr.f - row0 * res,
                    "n_active_px": n_active, "n_target_px": n_target,
                    "tile_clipped": clipped,
                })
    return pd.DataFrame(rows)


def main() -> None:
    from pipeline.hrrr import pick_mvp_fires

    p = argparse.ArgumentParser(description="Phase 3 feature stack and sample index.")
    p.add_argument("what", choices=["static", "index", "hrrr-check"])
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--events", default="data/processed/fire_events.csv")
    p.add_argument("--fire-id")
    p.add_argument("--all-mvp", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config)
    fires = pick_mvp_fires(ROOT / a.events)
    if a.fire_id:
        fires = fires[fires["fire_id"] == a.fire_id]
    elif not a.all_mvp:
        raise SystemExit("pass --fire-id or --all-mvp")
    ids = fires["fire_id"].tolist()

    if a.what == "static":
        (ROOT / "data/processed/features").mkdir(parents=True, exist_ok=True)
        for fid in ids:
            r = build_static(fid, cfg, a.overwrite)
            if r["status"] == "built":
                print(f"{fid}  {r['shape']}  elev {r['elev_m']} m  slope p95 {r['slope_deg_p95']}deg  "
                      f"{r['fuel_classes']} fuel classes  CC<={r['cc_max']}%  CH<={r['ch_max_m']} m  "
                      f"{r['mb']} MB")
            else:
                print(f"{fid}  {r['status']}")

    elif a.what == "hrrr-check":
        rows = [hrrr_coverage(f, cfg) for f in ids]
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print(f"\nweather features need {df.hours_needed.sum():,} hours; "
              f"{df.hours_present.sum():,} present ({100*df.hours_present.sum()/df.hours_needed.sum():.1f}%)")

    else:
        parts = [build_index(f, cfg) for f in ids]
        idx = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        out = ROOT / "data/processed/sample_index.parquet"
        idx.to_parquet(out, index=False)
        per = idx.groupby("fire_id").agg(
            samples=("t_index", "size"), timesteps=("t_index", "nunique"),
            clipped=("tile_clipped", "sum"))
        per["tiles_per_step"] = (per["samples"] / per["timesteps"]).round(2)
        print(per.to_string())
        print(f"\n{len(idx):,} samples over {idx.fire_id.nunique()} fires "
              f"({idx.tile_clipped.sum():,} with clipped targets)")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
