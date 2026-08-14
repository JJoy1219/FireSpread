"""DEM ingestion (Phase 1.4).

Source is USGS 3DEP 1/3 arc-second (~10 m) on AWS Open Data — public, no credentials,
and the "3DEP tiles" option named in the CLAUDE.md data tree. Tiles are Cloud Optimized
GeoTIFFs, so `/vsicurl/` windowed reads pull only the bytes covering a fire instead of
the whole 468 MB tile. Statewide at 10 m would be ~25 GB; per-fire clips are a few tens
of MB each.

Elevation is stored raw (native EPSG:4269, native resolution) per CLAUDE.md Phase 1.4 —
slope, aspect and TPI are derived later in the feature pipeline, not stored here.

    python -m pipeline.dem --all-mvp
    python -m pipeline.dem --fire-id 2018_4037
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

# Must be set before rasterio imports GDAL, or the HTTP reads list whole buckets.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.merge import merge

from pipeline.download import ROOT, load_config

BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
TO_WGS = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)


def tile_name(lat: float, lon: float) -> str:
    """3DEP tiles are named for their NW corner: n40w122 spans 39-40N, 122-121W."""
    return f"n{math.floor(lat) + 1:02d}w{abs(math.floor(lon)):03d}"


def tiles_for_bbox(w: float, s: float, e: float, n: float) -> list[str]:
    names = []
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            names.append(tile_name(lat + 0.5, lon + 0.5))
    return sorted(set(names))


def url_for(tile: str) -> str:
    return f"/vsicurl/{BASE}/{tile}/USGS_13_{tile}.tif"


def fetch_dem(bbox_wgs84: tuple[float, float, float, float], dest: Path,
              overwrite: bool = False) -> Path | None:
    """Clip 3DEP elevation to a WGS84 bbox, mosaicking tiles where a fire spans one."""
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return dest

    w, s, e, n = bbox_wgs84
    names = tiles_for_bbox(w, s, e, n)
    srcs, missing = [], []
    for t in names:
        try:
            srcs.append(rasterio.open(url_for(t)))
        except rasterio.RasterioIOError:
            missing.append(t)
    if missing:
        print(f"    tiles unavailable: {missing}")
    if not srcs:
        return None

    # merge() does windowed reads against the COGs, so only the bbox travels the wire.
    arr, transform = merge(srcs, bounds=(w, s, e, n))
    profile = srcs[0].profile
    profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform,
                   count=1, compress="deflate", predictor=3, tiled=True)
    for src in srcs:
        src.close()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(arr[0], 1)
    return dest


def bbox_for_fire(row: pd.Series, margin_m: float = 20000) -> tuple[float, float, float, float]:
    """Fire extent in EPSG:5070 plus margin, converted to WGS84.

    The margin covers the 25.6 km patch footprint around perimeter tiles plus the
    neighbourhood that slope and the 300 m-radius TPI need at the patch edge.
    """
    x0, x1 = row["x_min"] - margin_m, row["x_max"] + margin_m
    y0, y1 = row["y_min"] - margin_m, row["y_max"] + margin_m
    lons, lats = TO_WGS.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return min(lons), min(lats), max(lons), max(lats)


def main() -> None:
    from pipeline.hrrr import pick_mvp_fires

    p = argparse.ArgumentParser(description="Download 3DEP elevation clips per fire.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--events", default="data/processed/fire_events.csv")
    p.add_argument("--fire-id")
    p.add_argument("--all-mvp", action="store_true")
    p.add_argument("--margin-km", type=float, default=20.0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = ROOT / cfg["paths"]["raw_dem"]
    fires = pick_mvp_fires(ROOT / args.events)
    if args.fire_id:
        fires = fires[fires["fire_id"] == args.fire_id]
    elif not args.all_mvp:
        raise SystemExit("pass --fire-id or --all-mvp")

    total = 0
    for _, f in fires.iterrows():
        bbox = bbox_for_fire(f, args.margin_km * 1000)
        tiles = tiles_for_bbox(*bbox)
        dest = out_root / f"{f['fire_id']}_dem.tif"
        print(f"{f['fire_id']}: bbox {bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}"
              f"  tiles {tiles}")
        r = fetch_dem(bbox, dest, args.overwrite)
        if r is None:
            print("    FAILED")
            continue
        mb = r.stat().st_size / 1e6
        total += mb
        with rasterio.open(r) as ds:
            a = ds.read(1, masked=True)
            print(f"    {ds.width} x {ds.height}  {mb:.1f} MB  "
                  f"elev {a.min():.0f}-{a.max():.0f} m")
    print(f"\ntotal {total:.1f} MB in {out_root}")


if __name__ == "__main__":
    main()
