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


def build_statewide(cfg: dict, res: float = 100.0, overwrite: bool = False,
                    margin_km: float = 30.0, attempts: int = 3) -> Path:
    """One statewide DEM on the modelling grid: EPSG:5070, 100 m.

    Per-fire 10 m clips do not scale — 1.16 GB for five fires is ~144 GB of transfer
    across 622, for data that is immediately resampled to 100 m anyway. Instead each
    3DEP tile is read **decimated**, which serves the request from the COG's overview
    pyramid rather than full resolution, then reprojected into a single statewide grid.
    Transfer drops to a few hundred MB, fetched once.

    The grid origin is snapped to a multiple of `res`, and the label rasters are already
    on 100 m EPSG:5070 origins, so per-fire crops land exactly on cell boundaries and the
    later warp is a crop rather than an interpolation.
    """
    import numpy as np
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    dest = ROOT / cfg["paths"]["raw_dem"] / f"california_{int(res)}m_5070.tif"
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return dest

    # Pad beyond the region bbox: a fire's raster carries a 20 km margin for the patch
    # footprint and TPI neighbourhood, so fires near the border (Mexico, Oregon, Nevada,
    # the coast) need elevation *outside* California. Without this, 47 fires came out
    # with holes and one at 3.5% coverage.
    w, s, e, n = cfg["region"]["bbox"]
    dlat = margin_km / 111.0
    dlon = margin_km / (111.0 * math.cos(math.radians((s + n) / 2)))
    w, s, e, n = w - dlon, s - dlat, e + dlon, n + dlat
    to_5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = to_5070.transform([w, e, w, e], [s, s, n, n])
    x0 = math.floor(min(xs) / res) * res
    y1 = math.ceil(max(ys) / res) * res
    width = int(math.ceil((max(xs) - x0) / res))
    height = int(math.ceil((y1 - min(ys)) / res))
    transform = from_origin(x0, y1, res, res)
    print(f"statewide grid {width} x {height} @ {res:.0f} m EPSG:5070 "
          f"({width*height*4/1e6:.0f} MB in memory)")

    out = np.full((height, width), np.nan, dtype="float32")
    tiles = tiles_for_bbox(w, s, e, n)
    print(f"{len(tiles)} 3DEP tiles")

    def add_tile(name: str) -> bool:
        with rasterio.open(url_for(name)) as src:
            # Decimate on read: GDAL serves this from the overview pyramid, so a 1°
            # tile costs a few MB instead of ~400.
            factor = max(int(round(res / (abs(src.transform.a) * 111320))), 1)
            oh, ow = max(src.height // factor, 1), max(src.width // factor, 1)
            arr = src.read(1, out_shape=(1, oh, ow), resampling=Resampling.average,
                           masked=True)
            src_tr = src.transform * src.transform.scale(src.width / ow, src.height / oh)
            buf = np.full((height, width), np.nan, dtype="float32")
            reproject(source=arr.filled(np.nan), destination=buf,
                      src_transform=src_tr, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:5070",
                      src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            m = np.isfinite(buf)
            out[m] = buf[m]
        return True

    done, failed_tiles = 0, []
    for i, t in enumerate(tiles, 1):
        # Retry: the first build lost 21 of 110 tiles to transient vsicurl errors while
        # other downloads were saturating the link, and every one of them opened fine
        # immediately afterwards. A dropped tile is a silent hole in the DEM, so it must
        # not be accepted on a single failure.
        for attempt in range(attempts):
            try:
                add_tile(t)
                done += 1
                break
            except rasterio.RasterioIOError:
                if attempt == attempts - 1:
                    failed_tiles.append(t)
        if i % 20 == 0 or i == len(tiles):
            print(f"  {i:>3}/{len(tiles)} tiles  {100*np.isfinite(out).mean():5.1f}% covered",
                  flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", driver="GTiff", height=height, width=width, count=1,
                       dtype="float32", crs="EPSG:5070", transform=transform,
                       nodata=np.nan, compress="deflate", predictor=3, tiled=True) as dst:
        dst.write(out, 1)
    finite = np.isfinite(out)
    print(f"\n{done} tiles merged, {len(failed_tiles)} unavailable after {attempts} attempts, "
          f"{100*finite.mean():.1f}% covered")
    if failed_tiles:
        print(f"  missing tiles: {failed_tiles}")
    print(f"elev {np.nanmin(out):.0f}-{np.nanmax(out):.0f} m, "
          f"{dest.stat().st_size/1e6:.0f} MB -> {dest}")
    return dest


def main() -> None:
    from pipeline.hrrr import pick_mvp_fires

    p = argparse.ArgumentParser(description="Download 3DEP elevation clips per fire.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--events", default="data/processed/fire_events.csv")
    p.add_argument("--fire-id")
    p.add_argument("--all-mvp", action="store_true")
    p.add_argument("--statewide", action="store_true",
                   help="build the statewide 100 m EPSG:5070 DEM (the default source)")
    p.add_argument("--margin-km", type=float, default=20.0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.statewide:
        build_statewide(cfg, overwrite=args.overwrite)
        return
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
