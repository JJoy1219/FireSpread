"""Feature stack construction and the tiled sample index (Phase 3).

Everything is resampled onto **the exact grid the labels already use** — the transform
and shape are read back from each fire's label zarr rather than recomputed, so features
and targets cannot drift apart by a half pixel.

Two halves:

* `build_static` — terrain and fuel. Static per fire, so computed once.
* `build_index`  — enumerates (fire_id, timestep, tile) samples over the active
  perimeter, which is the option-A tiling scheme (see README).

* `weather_tile` — regrids the stored HRRR window onto a tile of that same grid and
  derives Fosberg fuel moisture. Computed on demand rather than materialised: a
  per-fire weather raster would be ~2 GB for Creek alone, against 23.8 MB for all
  five fires in native HRRR resolution.

    python -m pipeline.features static  --all-mvp
    python -m pipeline.features index   --all-mvp
    python -m pipeline.features weather --all-mvp   # validate the weather path
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
from pyproj import Transformer
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

# HRRR CONUS Lambert Conformal. Verified rather than assumed: projecting the stored
# per-cell lon/lat through this CRS reproduces a regular 3000 m grid to within 0.7 m,
# which is float32 coordinate precision. `hrrr_affine` re-checks it per fire.
HRRR_LCC = ("+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 "
            "+x_0=0 +y_0=0 +R=6371229 +units=m +no_defs")
WEATHER_CHANNELS = ["u10_lag0", "u10_lag6", "u10_lag12",
                    "v10_lag0", "v10_lag6", "v10_lag12",
                    "rh2m", "t2m", "fosberg_10h"]


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


def hrrr_affine(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float, float, float]:
    """Fit the HRRR window's regular grid in Lambert coordinates: (x0, y0, dx, dy).

    Raises if the fit is not regular, which is the check that `HRRR_LCC` is the right
    projection — a wrong CRS shows up immediately as a non-constant step.
    """
    tf = Transformer.from_crs("EPSG:4326", HRRR_LCC, always_xy=True)
    x, y = tf.transform(lon, lat)
    dx = float(np.mean(np.diff(x, axis=1)))
    dy = float(np.mean(np.diff(y, axis=0)))
    resid = max(float(np.abs(np.diff(x, axis=1) - dx).max()),
                float(np.abs(np.diff(y, axis=0) - dy).max()))
    if resid > 5.0 or not (2000 < abs(dx) < 4000):
        raise SystemExit(f"HRRR window is not regular in {HRRR_LCC!r} "
                         f"(step {dx:.1f}/{dy:.1f} m, residual {resid:.2f} m)")
    return float(x[0, 0]), float(y[0, 0]), dx, dy


def hrrr_index_grid(fire_id: str, cfg: dict, row0: int, col0: int,
                    patch: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (row, col) into the HRRR window for every cell of one tile.

    Fire grid -> EPSG:5070 metres -> lon/lat -> HRRR Lambert -> fractional index.
    Done per tile rather than per fire because regridding a whole fire at 100 m for
    all 9 wind times would cost ~120 MB per sample for a fire the size of Creek.
    """
    tr, H, W, crs = label_grid(fire_id)
    g = zarr.open_group(ROOT / cfg["paths"]["hrrr_windows"] / f"{fire_id}.zarr", mode="r")
    x0, y0, dx, dy = hrrr_affine(np.asarray(g["lon"]), np.asarray(g["lat"]))

    rows = np.arange(row0, row0 + patch) + 0.5
    cols = np.arange(col0, col0 + patch) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    xs, ys = tr * (cc, rr)                       # EPSG:5070 cell centres

    tf = Transformer.from_crs(crs, HRRR_LCC, always_xy=True)
    hx, hy = tf.transform(xs, ys)
    return (hy - y0) / dy, (hx - x0) / dx


def bilinear(cube: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Sample `(C, ny, nx)` at fractional indices, clamped at the window edge."""
    ny, nx = cube.shape[-2:]
    r = np.clip(r, 0, ny - 1.001)
    c = np.clip(c, 0, nx - 1.001)
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    fr, fc = (r - r0)[None], (c - c0)[None]
    return (cube[:, r0, c0] * (1 - fr) * (1 - fc) + cube[:, r0, c0 + 1] * (1 - fr) * fc
            + cube[:, r0 + 1, c0] * fr * (1 - fc) + cube[:, r0 + 1, c0 + 1] * fr * fc)


def fosberg_10h(t_k: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """10-h dead fuel moisture from the Fosberg equilibrium relations.

    Piecewise in RH with a temperature correction, defined in Fahrenheit, so the
    kelvin HRRR field is converted here. Computed *after* regridding: the relation is
    nonlinear, so evaluating it on the 3 km grid and interpolating the result is not
    the same as interpolating the inputs, and would smear the dry extremes that matter.
    """
    t_f = (t_k - 273.15) * 9.0 / 5.0 + 32.0
    rh = np.clip(rh, 0.0, 100.0)
    m = np.where(
        rh < 10,
        0.03229 + 0.281073 * rh - 0.000578 * rh * t_f,
        np.where(
            rh < 50,
            2.22749 + 0.160107 * rh - 0.014784 * t_f,
            21.0606 + 0.005565 * rh**2 - 0.00035 * rh * t_f - 0.483199 * rh,
        ),
    )
    # 1-h equilibrium -> 10-h timelag: the standard NFDRS damping toward equilibrium.
    return np.clip(m * 1.28, 1.0, 40.0).astype("float32")


def _hour_cube(fire_id: str, cfg: dict) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    g = zarr.open_group(ROOT / cfg["paths"]["hrrr_windows"] / f"{fire_id}.zarr", mode="r")
    filled = np.asarray(g["filled"])
    return np.asarray(g["data"]), {s: i for i, s in enumerate(g.attrs["times"])}, filled


def _hour_field(data, pos, filled, t: pd.Timestamp) -> np.ndarray | None:
    """One hour's `(4, ny, nx)` field, interpolating an isolated gap from +/-1 h.

    The gap rule from Phase 1.2. The fetcher guarantees a missing hour has its +/-1 h
    neighbours present, so this interpolates across 2 h rather than the 12 h a 6-hourly
    set would otherwise force. Returns None when the hole is too wide to bridge, and
    the caller drops the sample.
    """
    i = pos.get(f"{t:%Y%m%d_%H}z")
    if i is not None and filled[i]:
        return data[i]
    lo = hi = None
    for dh in range(1, 4):
        j = pos.get(f"{t - pd.Timedelta(hours=dh):%Y%m%d_%H}z")
        if lo is None and j is not None and filled[j]:
            lo = (dh, data[j])
        k = pos.get(f"{t + pd.Timedelta(hours=dh):%Y%m%d_%H}z")
        if hi is None and k is not None and filled[k]:
            hi = (dh, data[k])
    if lo is None or hi is None:
        return None
    w = lo[0] / (lo[0] + hi[0])
    return lo[1] * (1 - w) + hi[1] * w


def weather_tile(fire_id: str, t_index: int, cfg: dict, row0: int, col0: int,
                 patch: int | None = None) -> np.ndarray | None:
    """`(t_steps, len(WEATHER_CHANNELS), patch, patch)` for one sample.

    Step k is the label window ending at T - k*window_hours, and its wind lags are
    relative to that step's own time -- so the sample spans
    window_hours*(t_steps-1) + 12 h of weather. See `model.channels` in the config.
    """
    patch = patch or int(cfg["grid"]["patch_size"])
    g = zarr.open_group(ROOT / "data/processed/labels" / f"{fire_id}.zarr", mode="r")
    times = [pd.Timestamp(t) for t in g.attrs["window_times"]]
    window_h = int(g.attrs["window_hours"])
    t_steps = int(cfg["model"]["t_steps"])

    data, pos, filled = _hour_cube(fire_id, cfg)
    r, c = hrrr_index_grid(fire_id, cfg, row0, col0, patch)

    out = np.zeros((t_steps, len(WEATHER_CHANNELS), patch, patch), dtype="float32")
    for k in range(t_steps):
        base = times[t_index] - pd.Timedelta(hours=window_h * k)
        lags = []
        for lag in (0, 6, 12):
            f = _hour_field(data, pos, filled, base - pd.Timedelta(hours=lag))
            if f is None:
                return None
            lags.append(bilinear(f, r, c))       # (4, patch, patch): u, v, t2m, rh
        step = t_steps - 1 - k                   # oldest first along the sequence axis
        out[step, 0:3] = [l[0] for l in lags]    # u10 at lag 0, 6, 12
        out[step, 3:6] = [l[1] for l in lags]    # v10
        out[step, 6] = lags[0][3]                # rh at this step's own time
        out[step, 7] = lags[0][2]                # t2m
        out[step, 8] = fosberg_10h(lags[0][2], lags[0][3])
    return out


def static_tile(fire_id: str, cfg: dict, row0: int, col0: int,
                patch: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Static channels and the fuel index for one tile, honouring `elevation_mode`.

    `fire_centred` subtracts the fire's own mean elevation. Measured over the 661 MVP
    tiles, 59.3% of elevation's variance is *between* tiles rather than within them --
    far the highest of any static channel -- so its absolute level acts as a per-fire
    identity handle. Centring keeps the within-patch gradient, including the km-scale
    rise that the 300 m TPI cannot see, and drops the identity component. The mean is
    taken over the whole fire, not the tile, so overlapping tiles stay consistent.
    """
    patch = patch or int(cfg["grid"]["patch_size"])
    g = zarr.open_group(ROOT / "data/processed/features" / f"{fire_id}.zarr", mode="r")
    chans = list(g.attrs["channels"])
    sl = (slice(row0, row0 + patch), slice(col0, col0 + patch))
    stack = np.asarray(g["static"][:, sl[0], sl[1]]).astype("float32")
    fuel = np.asarray(g["fuel"][sl]).astype("int16")

    mode = str(cfg["model"].get("elevation_mode", "fire_centred"))
    ei = chans.index("elevation")
    if mode == "fire_centred":
        stack[ei] -= float(np.asarray(g["static"][ei]).mean())
    elif mode == "none":
        stack = np.delete(stack, ei, axis=0)
    elif mode != "raw":
        raise SystemExit(f"unknown model.elevation_mode: {mode!r}")
    return stack, fuel


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
    p.add_argument("what", choices=["static", "index", "hrrr-check", "weather"])
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

    elif a.what == "weather":
        # Weather is never materialised, so "building" it means proving every sample in
        # the index can produce a finite, physical tensor on demand.
        idx = pd.read_parquet(ROOT / "data/processed/sample_index.parquet")
        rows = []
        for fid in ids:
            sub = idx[idx["fire_id"] == fid]
            ok = dropped = 0
            stats = []
            for _, s in sub.iterrows():
                w = weather_tile(fid, int(s["t_index"]), cfg, int(s["row0"]), int(s["col0"]))
                if w is None:
                    dropped += 1
                    continue
                if not np.isfinite(w).all():
                    raise SystemExit(f"{fid} t={s['t_index']} produced non-finite weather")
                ok += 1
                # Mean wind *speed*, not the speed of the mean vector -- averaging u and
                # v first cancels opposing directions and reports a near-zero breeze.
                stats.append([float(np.hypot(w[:, 0], w[:, 3]).mean()),
                              float(w[:, 6].mean()), float(w[:, 7].mean()),
                              float(w[:, 8].mean())])
            m = np.array(stats).mean(axis=0)
            rows.append({"fire_id": fid, "samples": len(sub), "ok": ok, "dropped": dropped,
                         "wind_ms": round(m[0], 1), "rh_pct": round(m[1], 1),
                         "t_c": round(m[2] - 273.15, 1), "fosberg_pct": round(m[3], 1)})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print(f"\n{df.ok.sum():,}/{df.samples.sum():,} samples produce weather "
              f"({df.dropped.sum()} dropped on unbridgeable gaps)")

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
