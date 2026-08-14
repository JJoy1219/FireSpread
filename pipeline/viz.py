"""Visualisation helpers. Not in the CLAUDE.md tree, but the stack lists matplotlib
for exactly this and the pipeline needs eyeballing at every stage.

    python -m pipeline.viz spread --fire-id 2018_4037
    python -m pipeline.viz panel
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize
from pyproj import Transformer

from pipeline.download import ROOT, load_config

TF = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)


def load_fire(fire_id: str) -> pd.DataFrame:
    det = pd.read_parquet(ROOT / "data/processed/detections_labeled.parquet")
    d = det[det["fire_id"] == fire_id].copy()
    if d.empty:
        raise SystemExit(f"no detections for {fire_id}")
    d["dt"] = pd.to_datetime(d["acq_datetime"])
    d["hours"] = (d["dt"] - d["dt"].min()).dt.total_seconds() / 3600
    return d.sort_values("dt")


def hrrr_wind(fire_id: str, when: datetime, bbox: tuple[float, float, float, float]):
    """Sample the HRRR 10 m wind field over a projected bounding box."""
    f = ROOT / "data/raw/hrrr" / fire_id / f"{when:%Y%m%d_%H}z.grib2"
    if not f.exists():
        return None
    ds = xr.open_dataset(
        f, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10},
                        "indexpath": ""},
    )
    lat = ds.latitude.values
    lon = ds.longitude.values - 360
    x, y = TF.transform(lon, lat)
    x0, x1, y0, y1 = bbox
    pad = 0.15 * max(x1 - x0, y1 - y0)
    m = (x >= x0 - pad) & (x <= x1 + pad) & (y >= y0 - pad) & (y <= y1 + pad)
    if not m.any():
        ds.close()
        return None
    out = (x[m], y[m], ds.u10.values[m], ds.v10.values[m])
    ds.close()
    return out


def plot_spread(fire_id: str, title: str, out: Path, wind_at_hour: int | None = 12) -> Path:
    d = load_fire(fire_id)
    x0, x1 = d.x.min(), d.x.max()
    y0, y1 = d.y.min(), d.y.max()
    pad = 0.08 * max(x1 - x0, y1 - y0, 5000)
    bbox = (x0, x1, y0, y1)

    fig, ax = plt.subplots(figsize=(9.5, 8.5), dpi=130)

    w = None
    if wind_at_hour is not None:
        w = hrrr_wind(fire_id, (d.dt.min() + timedelta(hours=wind_at_hour)).to_pydatetime(), bbox)
    if w is not None:
        wx, wy, u, v = w
        spd = np.hypot(u, v)
        ax.quiver(wx, wy, u, v, spd, cmap="Blues", alpha=0.55, scale=260, width=0.0034,
                  pivot="mid", zorder=1)

    # Detections coloured by hours since first detection = the observed spread.
    sc = ax.scatter(d.x, d.y, c=d.hours, s=13, cmap="inferno_r", alpha=0.85,
                    norm=Normalize(0, d.hours.max()), zorder=3, linewidths=0)
    ax.scatter(d.x.iloc[0], d.y.iloc[0], marker="*", s=420, facecolor="#00e5ff",
               edgecolor="black", linewidth=1.1, zorder=5, label="first detection")

    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("hours since first detection", fontsize=10)

    # 10 km scale bar in projected metres.
    sx = x0 - pad + 0.05 * (x1 - x0 + 2 * pad)
    sy = y0 - pad + 0.05 * (y1 - y0 + 2 * pad)
    ax.plot([sx, sx + 10000], [sy, sy], color="black", lw=3, zorder=6)
    ax.text(sx + 5000, sy + 0.012 * (y1 - y0 + 2 * pad), "10 km", ha="center", fontsize=9)

    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    sub = f"{len(d):,} VIIRS detections  |  {d.hours.max():.0f} h  |  EPSG:5070, 100 m modelling grid"
    if w is not None:
        sub += f"\nblue arrows: HRRR 10 m wind at T+{wind_at_hour} h"
    ax.set_title(f"{title}\n{sub}", fontsize=12, loc="left")
    ax.set_xlabel("EPSG:5070 easting (m)")
    ax.set_ylabel("EPSG:5070 northing (m)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15, linestyle=":")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_panel(fires: dict[str, str], out: Path) -> Path:
    """Small multiples: every MVP fire at the same physical scale."""
    n = len(fires)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.4), dpi=130)
    for ax, (fid, name) in zip(np.atleast_1d(axes), fires.items()):
        d = load_fire(fid)
        cx, cy = (d.x.min() + d.x.max()) / 2, (d.y.min() + d.y.max()) / 2
        ax.scatter(d.x - cx, d.y - cy, c=d.hours, s=3.5, cmap="inferno_r", alpha=0.85, linewidths=0)
        half = 30000  # same 60 km window for every fire, so sizes are comparable
        ax.add_patch(plt.Rectangle((-12800, -12800), 25600, 25600, fill=False,
                                   edgecolor="#00e5ff", lw=1.3, ls="--", zorder=4))
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
        ax.set_title(f"{name}\n{len(d):,} det  |  {d.hours.max()/24:.0f} d", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MVP fires at identical scale — dashed box is one 256x256 patch (25.6 km)",
                 fontsize=12, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_layers(fire_id: str, title: str, out: Path) -> Path:
    """Elevation, fuel model and burn progression on one EPSG:5070 extent.

    This is the co-registration test: if FIRMS, LANDFIRE and 3DEP disagree about
    where anything is, the detections will not sit on the terrain they burned.
    """
    import glob
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    d = load_fire(fire_id)
    x0, x1 = d.x.min(), d.x.max()
    y0, y1 = d.y.min(), d.y.max()
    pad = 0.10 * max(x1 - x0, y1 - y0)
    ext = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)

    # DEM arrives in EPSG:4269; warp it onto the modelling CRS for display.
    dem_p = ROOT / "data/raw/dem" / f"{fire_id}_dem.tif"
    with rasterio.open(dem_p) as src:
        tr, w, h = calculate_default_transform(
            src.crs, "EPSG:5070", src.width, src.height, *src.bounds, resolution=100)
        dem = np.empty((h, w), dtype="float32")
        reproject(rasterio.band(src, 1), dem, dst_transform=tr, dst_crs="EPSG:5070",
                  resampling=Resampling.bilinear)
    dem_ext = (tr.c, tr.c + w * 100, tr.f - h * 100, tr.f)
    dem = np.where(dem < -1000, np.nan, dem)

    lf_p = sorted(glob.glob(str(ROOT / "data/raw/landfire" / f"{fire_id}_*.tif")))[0]
    with rasterio.open(lf_p) as src:
        fuel = src.read(1).astype("float32")
        b = src.bounds
        lf_ext = (b.left, b.right, b.bottom, b.top)
    # Scott-Burgan 40 collapsed to broad fuel types for a readable figure.
    bins = [0, 100, 110, 130, 150, 170, 190, 210]
    labels = ["non-burnable", "grass", "grass-shrub", "shrub",
              "timber-understory", "timber-litter", "slash"]
    fuel_c = np.digitize(fuel, bins) - 1
    fuel_c = np.where(fuel < 1, np.nan, fuel_c)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), dpi=125)
    im0 = axes[0].imshow(dem, extent=dem_ext, origin="upper", cmap="terrain")
    axes[0].set_title("3DEP elevation (m), warped to EPSG:5070")
    fig.colorbar(im0, ax=axes[0], shrink=0.75)

    cmap = plt.get_cmap("Set2", len(labels))
    im1 = axes[1].imshow(fuel_c, extent=lf_ext, origin="upper", cmap=cmap,
                         vmin=-0.5, vmax=len(labels) - 0.5)
    axes[1].set_title(f"LANDFIRE {Path(lf_p).stem.split('_')[-1]} FBFM40 fuel type")
    cb = fig.colorbar(im1, ax=axes[1], shrink=0.75, ticks=range(len(labels)))
    cb.ax.set_yticklabels(labels, fontsize=8)

    axes[2].imshow(dem, extent=dem_ext, origin="upper", cmap="gray", alpha=0.55)
    sc = axes[2].scatter(d.x, d.y, c=d.hours, s=5, cmap="inferno_r", linewidths=0)
    axes[2].set_title("FIRMS burn progression over terrain")
    fig.colorbar(sc, ax=axes[2], shrink=0.75, label="hours since first detection")

    for ax in axes:
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title} — three sources, one EPSG:5070 extent (co-registration check)",
                 fontsize=13, y=0.99)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_labels(fire_id: str, title: str, out: Path, n_panels: int = 4,
                horizon_steps: int = 1) -> Path:
    """Show real (label, target) training pairs: burned-so-far vs the next 6 h of growth."""
    import zarr

    g = zarr.open_group(ROOT / "data/processed/labels" / f"{fire_id}.zarr", mode="r")
    burn = g["burn_new"]
    times = [pd.Timestamp(t) for t in g.attrs["window_times"]]
    usable = np.array(g.attrs["usable"], dtype=bool)

    new_px = np.array([int(burn[i].sum()) for i in range(burn.shape[0])])
    # Pick the most active usable windows — that is where the model earns its keep.
    ok = np.where(usable[:-horizon_steps])[0]
    picks = sorted(ok[np.argsort(new_px[ok])[-n_panels:]])

    cum = np.cumsum(np.stack([burn[i] for i in range(burn.shape[0])]), axis=0) > 0

    fig, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 5.0), dpi=130)
    for ax, t in zip(np.atleast_1d(axes), picks):
        label = cum[t]
        target = np.zeros_like(label)
        for k in range(1, horizon_steps + 1):
            if t + k < burn.shape[0]:
                target |= (burn[t + k] > 0)
        target &= ~label                       # only genuinely new burning counts

        rgb = np.zeros((*label.shape, 3), dtype=float)
        rgb[label] = [0.55, 0.20, 0.10]        # burned so far  (model input)
        rgb[target] = [1.00, 0.85, 0.10]       # next 6 h       (model target)
        ax.imshow(rgb)
        ax.set_title(f"{times[t]:%Y-%m-%d %H:%M}Z\nburned {label.sum():,} px  "
                     f"-> +{target.sum():,} px", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    hz = int(g.attrs["window_hours"]) * horizon_steps
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=(0.55, 0.20, 0.10), label="label: burned up to T"),
                        Patch(color=(1.0, 0.85, 0.10), label=f"target: new burn T to T+{hz}h")],
               loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(f"{title} — actual training pairs at 100 m, EPSG:5070", fontsize=13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


MVP = {
    "2017_2405": "2017 Sierra (median)",
    "2018_4037": "Camp (2018)",
    "2020_3779": "Creek (2020)",
    "2021_3526": "Caldor (2021)",
    "2022_3298": "Mosquito (2022)",
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["spread", "panel", "layers"])
    p.add_argument("--fire-id", default="2018_4037")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    outdir = ROOT / "figures"
    if a.what == "spread":
        f = plot_spread(a.fire_id, a.title or MVP.get(a.fire_id, a.fire_id),
                        Path(a.out) if a.out else outdir / f"spread_{a.fire_id}.png")
    elif a.what == "layers":
        f = plot_layers(a.fire_id, a.title or MVP.get(a.fire_id, a.fire_id),
                        Path(a.out) if a.out else outdir / f"layers_{a.fire_id}.png")
    else:
        f = plot_panel(MVP, Path(a.out) if a.out else outdir / "mvp_panel.png")
    print(f"wrote {f}")
