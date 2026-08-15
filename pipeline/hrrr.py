"""HRRR weather ingestion (Phase 1.2).

HRRR surface analysis files on `s3://noaa-hrrr-bdp-pds/` are ~130 MB each, but the
four fields we need are four GRIB2 messages totalling a few MB. Every file has a
sibling `.idx` listing each message's byte offset, so we fetch only the ranges we
want with HTTP Range requests. Concatenated GRIB2 messages are themselves a valid
GRIB2 file, so the subset opens directly in cfgrib.

    hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf00.grib2

Each fire's GRIBs are cropped to a local window and written into a per-fire zarr, then
deleted (`storage.keep_raw_hrrr_grib`). Retaining the GRIBs costs ~204 GB across the
archive; the windows cost ~4.5 GB.

Usage:
    python -m pipeline.hrrr --list-fires          # choose the MVP events
    python -m pipeline.hrrr --fire-id 2018_4037
    python -m pipeline.hrrr --all-mvp
    python -m pipeline.hrrr --all-mvp --dry-run   # hours and MB, fetch nothing
"""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import zarr
from botocore import UNSIGNED
from botocore.config import Config
from pyproj import Transformer

from pipeline.download import ROOT, load_config

BUCKET = "noaa-hrrr-bdp-pds"
# (short name in the .idx, level string in the .idx) -> our channel name
FIELDS = {
    ("UGRD", "10 m above ground"): "u10",
    ("VGRD", "10 m above ground"): "v10",
    ("TMP", "2 m above ground"): "t2m",
    ("RH", "2 m above ground"): "rh2m",
}
# Names cfgrib gives these once decoded — note RH decodes as `r2`, not `rh2m`.
CHANNELS = ["u10", "v10", "t2m", "r2"]
# Feature spec (CLAUDE.md Phase 3): wind at T, T-6h, T-12h. Temperature and RH at T
# only, but they ride along in the same messages so there is nothing to save by
# fetching them on a different schedule.
WIND_LAG_HOURS = (0, 6, 12)
# HRRR on this bucket begins here; earlier fires would need ERA5 instead.
HRRR_EPOCH = datetime(2014, 7, 30, tzinfo=timezone.utc)

_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5}))


def _key(dt: datetime, fxx: int = 0) -> str:
    return f"hrrr.{dt:%Y%m%d}/conus/hrrr.t{dt:%H}z.wrfsfcf{fxx:02d}.grib2"


def parse_idx(key: str) -> pd.DataFrame:
    """Fetch and parse the .idx sidecar into a frame with byte offsets."""
    body = _s3.get_object(Bucket=BUCKET, Key=key + ".idx")["Body"].read().decode()
    rows = []
    for line in body.strip().split("\n"):
        parts = line.split(":")
        rows.append({"msg": int(parts[0]), "start": int(parts[1]), "var": parts[3], "level": parts[4]})
    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    # A message ends where the next begins; the last one runs to EOF (-1 => open range).
    df["end"] = df["start"].shift(-1).fillna(0).astype("int64") - 1
    return df


def _contiguous_runs(sel: pd.DataFrame) -> list[tuple[int, int]]:
    """Merge adjacent messages into as few Range requests as possible."""
    runs: list[list[int]] = []
    for _, r in sel.sort_values("start").iterrows():
        if runs and r["start"] == runs[-1][1] + 1:
            runs[-1][1] = r["end"]
        else:
            runs.append([r["start"], r["end"]])
    return [(a, b) for a, b in runs]


def download_hour(dt: datetime, dest: Path, overwrite: bool = False) -> tuple[Path, int] | None:
    """Write a GRIB2 file holding just the four fields for one analysis hour."""
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return dest, dest.stat().st_size

    key = _key(dt)
    try:
        idx = parse_idx(key)
    except _s3.exceptions.NoSuchKey:
        print(f"  {dt:%Y-%m-%d %H}z  MISSING from archive")
        return None

    mask = pd.Series(False, index=idx.index)
    for (var, level) in FIELDS:
        mask |= (idx["var"] == var) & (idx["level"] == level)
    sel = idx[mask]
    if len(sel) != len(FIELDS):
        found = set(zip(sel["var"], sel["level"]))
        print(f"  {dt:%Y-%m-%d %H}z  expected {len(FIELDS)} messages, found {len(sel)}: {found}")
        return None

    blobs = []
    for start, end in _contiguous_runs(sel):
        rng = f"bytes={start}-" if end < start else f"bytes={start}-{end}"
        blobs.append(_s3.get_object(Bucket=BUCKET, Key=key, Range=rng)["Body"].read())

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"".join(blobs))
    return dest, dest.stat().st_size


def needed_hours(fire_id: str, cfg: dict) -> pd.DatetimeIndex:
    """The exact analysis hours the feature stack needs for one fire.

    Driven by the label zarr's own window times so the weather can never span a
    different period than the targets. For each usable window T the model sees
    `t_steps` windows ending at T, and each of those needs wind at its own
    T, T-6h, T-12h — so the needed set is

        {w - k*window_hours - lag : w in windows, k < t_steps, lag in WIND_LAG_HOURS}

    That is 3 hours per 24 h window, not the 4 a uniform 6-hourly grid would fetch:
    the 18h-offset hour is never read by any feature. Hence ~25% less transfer than
    the `storage.hrrr_step_hours` budget in the README assumed.
    """
    g = zarr.open_group(ROOT / "data/processed/labels" / f"{fire_id}.zarr", mode="r")
    window_h = int(g.attrs["window_hours"])
    usable = np.array(g.attrs["usable"], dtype=bool)
    times = [pd.Timestamp(t) for t in g.attrs["window_times"]]
    t_steps = int(cfg["model"]["t_steps"])

    wanted: set[pd.Timestamp] = set()
    for w, ok in zip(times, usable):
        if not ok:
            continue
        for k in range(t_steps):
            base = w - pd.Timedelta(hours=window_h * k)
            for lag in WIND_LAG_HOURS:
                wanted.add((base - pd.Timedelta(hours=lag)).floor("h"))
    return pd.DatetimeIndex(sorted(wanted))


def fire_window_bounds(fire_id: str, margin_km: float) -> tuple[float, float, float, float]:
    """Fire raster bounds in lon/lat, padded by `margin_km`."""
    g = zarr.open_group(ROOT / "data/processed/labels" / f"{fire_id}.zarr", mode="r")
    a, b, c, d, e, f = g.attrs["transform"][:6]
    _, H, W = g["burn_new"].shape
    xs = [c, c + a * W]
    ys = [f, f + e * H]
    tf = Transformer.from_crs(g.attrs.get("crs", "EPSG:5070"), "EPSG:4326", always_xy=True)
    corners = [tf.transform(x, y) for x in xs for y in ys]
    lons = [p[0] for p in corners]
    lats = [p[1] for p in corners]
    dlat = margin_km / 111.0
    dlon = margin_km / (111.0 * max(np.cos(np.radians(np.mean(lats))), 0.1))
    return min(lons) - dlon, min(lats) - dlat, max(lons) + dlon, max(lats) + dlat


def _grib_datasets(path: Path) -> list:
    import cfgrib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cfgrib.open_datasets(str(path))


def window_slice(path: Path, bounds: tuple[float, float, float, float]) -> tuple[slice, slice]:
    """Index window into the HRRR grid covering `bounds`.

    HRRR is Lambert Conformal with 2-D lat/lon, so a lon/lat box is not a rectangle in
    index space; we take the bounding box of every cell inside it. Longitudes come in
    the 0-360 convention and are folded to -180..180 first.
    """
    ds = _grib_datasets(path)[0]
    lat = np.asarray(ds["latitude"])
    lon = ((np.asarray(ds["longitude"]) + 180.0) % 360.0) - 180.0
    lon_min, lat_min, lon_max, lat_max = bounds
    inside = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if not inside.any():
        raise SystemExit(f"fire window {bounds} falls outside the HRRR domain")
    ys, xs = np.where(inside)
    return slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)


def window_lonlat(path: Path, sy: slice, sx: slice) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre lon/lat for the cropped window, folded to -180..180.

    Stored alongside the data because HRRR is Lambert Conformal: without the true
    per-cell coordinates there is no way to resample onto the 100 m EPSG:5070 fire
    grid, and interpolating the lon/lat bbox linearly is wrong by up to a cell.
    """
    ds = _grib_datasets(path)[0]
    lat = np.asarray(ds["latitude"])[sy, sx].astype("float32")
    lon = ((np.asarray(ds["longitude"])[sy, sx] + 180.0) % 360.0) - 180.0
    return lon.astype("float32"), lat


def extract_window(path: Path, sy: slice, sx: slice) -> np.ndarray:
    """Crop one hour's GRIB to `(len(CHANNELS), ny, nx)` float32."""
    found = {}
    for ds in _grib_datasets(path):
        for name in ds.data_vars:
            if name in CHANNELS:
                found[name] = np.asarray(ds[name])[sy, sx]
    missing = [c for c in CHANNELS if c not in found]
    if missing:
        raise SystemExit(f"{path.name}: GRIB decoded without {missing}; got {sorted(found)}")
    return np.stack([found[c] for c in CHANNELS]).astype("float32")


def _parse_stamp(s: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s[:-1], "%Y%m%d_%H"), tz="UTC")


def _sync_store(grp, stamps: list[str], bounds: tuple, overwrite: bool = False) -> None:
    """Make the store hold exactly `stamps`, carrying over rows already fetched.

    Needed because gap repair grows the hour list after the fact; rows are matched by
    timestamp, never by position, so re-syncing never silently reindexes data.
    """
    old_times = list(grp.attrs.get("times", []))
    if not overwrite and old_times == stamps and "filled" in grp:
        return

    new_filled = np.zeros(len(stamps), dtype=bool)
    if not overwrite and "data" in grp and "filled" in grp:
        old_filled = np.asarray(grp["filled"])
        old = grp["data"]
        pos = {s: i for i, s in enumerate(old_times)}
        buf = np.zeros((len(stamps), *old.shape[1:]), dtype="float32")
        for ni, s in enumerate(stamps):
            oi = pos.get(s)
            if oi is not None and oi < len(old_filled) and bool(old_filled[oi]):
                buf[ni] = old[oi]
                new_filled[ni] = True
        grp.create_array("data", shape=buf.shape, dtype="float32",
                         chunks=(1, *old.shape[1:]), overwrite=True)[:] = buf

    grp.create_array("filled", shape=(len(stamps),), dtype="bool",
                     overwrite=True)[:] = new_filled
    grp.attrs["times"] = stamps
    grp.attrs["channels"] = CHANNELS
    grp.attrs["bounds"] = list(bounds)


def _gap_neighbours(stamps: list[str], filled: np.ndarray) -> set[str]:
    """The +/-1 h hours bracketing every hour that came back missing.

    The stored set is 6-hourly, so without this an isolated hole would have to be
    interpolated across 12 h — far too long an assumption for 10 m wind. Two extra
    requests per hole buys a 2 h bracket instead.
    """
    out: set[str] = set()
    for s, f in zip(stamps, filled):
        if f:
            continue
        t = _parse_stamp(s)
        for dh in (-1, 1):
            out.add(f"{t + pd.Timedelta(hours=dh):%Y%m%d_%H}z")
    return out - set(stamps)


def download_event(
    fire_id: str,
    cfg: dict,
    overwrite: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    repair_rounds: int = 2,
) -> dict:
    """Fetch every hour the feature stack needs for one fire, window it, store it.

    Resumable: the zarr is allocated for the full needed span up front and a `filled`
    flag marks which hours are populated, so a rerun costs only what is still absent.
    Replaces the old fixed T-12h..T+24h window, which covered under 5% of what 24 h
    windows with `t_steps: 3` actually require.

    Any hour missing from the archive automatically pulls its +/-1 h neighbours so the
    gap can be interpolated across 2 h rather than 12. Bounded by `repair_rounds` so a
    long outage cannot walk outwards indefinitely.
    """
    hours = needed_hours(fire_id, cfg)
    if len(hours) == 0:
        return {"fire_id": fire_id, "error": "no usable label windows"}
    if hours[0].to_pydatetime().replace(tzinfo=timezone.utc) < HRRR_EPOCH:
        return {"fire_id": fire_id, "error": f"before HRRR epoch {HRRR_EPOCH:%Y-%m-%d}"}

    store = ROOT / cfg["paths"]["hrrr_windows"] / f"{fire_id}.zarr"
    bounds = fire_window_bounds(fire_id, float(cfg["storage"]["hrrr_window_margin_cells"]) * 3.0)

    if dry_run:
        return {"fire_id": fire_id, "hours_requested": len(hours), "hours_ok": 0,
                "missing": [], "mb": 0.0, "dir": str(store),
                "span": f"{hours[0]:%Y-%m-%d %H}z..{hours[-1]:%Y-%m-%d %H}z"}

    tmp_dir = ROOT / cfg["paths"]["raw_hrrr"] / fire_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    keep_grib = bool(cfg["storage"].get("keep_raw_hrrr_grib", False))

    grp = zarr.open_group(store, mode="a")
    stamps = [f"{t:%Y%m%d_%H}z" for t in hours]
    _sync_store(grp, stamps, bounds, overwrite)

    def fill(label: str = "") -> tuple[int, int]:
        """Fetch every unfilled hour currently in the store. Returns (ok, bytes)."""
        nonlocal stamps
        filled = grp["filled"]
        data = grp["data"] if "data" in grp and grp["data"].shape[0] == len(stamps) else None
        todo = [i for i in range(len(stamps)) if not bool(filled[i])]
        if limit is not None:
            todo = todo[:limit]

        sy = sx = None
        ok = nbytes = 0
        for n, i in enumerate(todo, 1):
            dt = _parse_stamp(stamps[i]).to_pydatetime()
            grib = tmp_dir / f"{stamps[i]}.grib2"
            res = download_hour(dt, grib, overwrite=True)
            if res is None:
                continue
            nbytes += res[1]

            if sy is None:
                sy, sx = window_slice(grib, bounds)
                if "lon" not in grp:
                    lon, lat = window_lonlat(grib, sy, sx)
                    grp.create_array("lon", shape=lon.shape, dtype="float32",
                                     overwrite=True)[:] = lon
                    grp.create_array("lat", shape=lat.shape, dtype="float32",
                                     overwrite=True)[:] = lat
            arr = extract_window(grib, sy, sx)
            if data is None:
                data = grp.create_array(
                    "data", shape=(len(stamps), *arr.shape), dtype="float32",
                    chunks=(1, len(CHANNELS), *arr.shape[1:]), overwrite=True,
                )
            data[i] = arr
            filled[i] = True
            ok += 1

            if not keep_grib:
                grib.unlink(missing_ok=True)
                for leftover in tmp_dir.glob(f"{stamps[i]}.grib2.*.idx"):
                    leftover.unlink(missing_ok=True)
            if n % 25 == 0 or n == len(todo):
                print(f"  {fire_id}{label}  {n:>4}/{len(todo)} hours  {nbytes/1e6:7.1f} MB fetched")
        return ok, nbytes

    ok, bytes_total = fill()

    # Any hour the archive does not have pulls its immediate neighbours, so Phase 3c
    # interpolates across 2 h instead of the 12 h the 6-hourly grid would otherwise force.
    repaired: list[str] = []
    for rnd in range(repair_rounds if limit is None else 0):
        extra = _gap_neighbours(stamps, np.asarray(grp["filled"]))
        if not extra:
            break
        print(f"  {fire_id}  gap repair round {rnd + 1}: fetching {len(extra)} neighbour hours")
        repaired.extend(sorted(extra))
        stamps = sorted(set(stamps) | extra, key=_parse_stamp)
        _sync_store(grp, stamps, bounds)
        r_ok, r_bytes = fill(" repair")
        ok += r_ok
        bytes_total += r_bytes

    filled = np.asarray(grp["filled"])
    missing = [s for s, f in zip(stamps, filled) if not f]
    grp.attrs["missing"] = missing
    grp.attrs["repair_hours"] = sorted(set(repaired) - set(missing))
    if not keep_grib and tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    stored = sum(p.stat().st_size for p in store.rglob("*") if p.is_file()) / 1e6
    return {
        "fire_id": fire_id,
        "hours_requested": len(stamps),
        "hours_ok": int(filled.sum()),
        "missing": missing,
        "repaired": grp.attrs["repair_hours"],
        "mb": round(bytes_total / 1e6, 1),
        "stored_mb": round(stored, 1),
        "dir": str(store),
    }


def download_all(fire_ids: list[str], cfg: dict, overwrite: bool = False,
                 dry_run: bool = False, repair_rounds: int = 2) -> dict:
    """Fetch every fire's hours, downloading each **calendar hour only once**.

    `download_event` is per fire, so fires burning in the same week re-fetch identical
    GRIBs. Across the full archive that redundancy is roughly 4x. Here the needed sets
    are unioned first, each unique hour is fetched once, and the message is windowed
    into every fire that wants it before the GRIB is deleted.

    Ordering matters for scratch: hours are processed in time order, so a GRIB is opened
    once, fanned out, and dropped — peak scratch stays one file regardless of fire count.
    """
    per_fire: dict[str, list[str]] = {}
    by_hour: dict[str, list[str]] = defaultdict(list)
    skipped = []
    for fid in fire_ids:
        try:
            hours = needed_hours(fid, cfg)
        except (KeyError, FileNotFoundError):
            skipped.append(fid)
            continue
        if len(hours) == 0 or hours[0].to_pydatetime().replace(tzinfo=timezone.utc) < HRRR_EPOCH:
            skipped.append(fid)
            continue
        stamps = [f"{t:%Y%m%d_%H}z" for t in hours]
        per_fire[fid] = stamps
        for s in stamps:
            by_hour[s].append(fid)

    fire_hours = sum(len(v) for v in per_fire.values())
    unique = len(by_hour)
    if dry_run:
        return {"fires": len(per_fire), "skipped": len(skipped), "fire_hours": fire_hours,
                "unique_hours": unique, "dedupe": round(fire_hours / max(unique, 1), 2),
                "est_gb": round(unique * 5.7 / 1000, 1)}

    root = ROOT / cfg["paths"]["hrrr_windows"]
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / cfg["paths"]["raw_hrrr"]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    keep_grib = bool(cfg["storage"].get("keep_raw_hrrr_grib", False))

    groups, bounds, slices, pos = {}, {}, {}, {}
    for fid, stamps in per_fire.items():
        g = zarr.open_group(root / f"{fid}.zarr", mode="a")
        b = fire_window_bounds(fid, float(cfg["storage"]["hrrr_window_margin_cells"]) * 3.0)
        _sync_store(g, stamps, b, overwrite)
        groups[fid], bounds[fid] = g, b
        pos[fid] = {s: i for i, s in enumerate(stamps)}

    missing_hours: list[str] = []
    ok = bytes_total = 0
    todo = sorted(by_hour)
    for n, stamp in enumerate(todo, 1):
        wanted = [f for f in by_hour[stamp] if not bool(groups[f]["filled"][pos[f][stamp]])]
        if not wanted:
            continue
        dt = _parse_stamp(stamp).to_pydatetime()
        grib = tmp_dir / f"{stamp}.grib2"
        res = download_hour(dt, grib, overwrite=True)
        if res is None:
            missing_hours.append(stamp)
            continue
        bytes_total += res[1]

        for fid in wanted:
            if fid not in slices:
                slices[fid] = window_slice(grib, bounds[fid])
                if "lon" not in groups[fid]:
                    lon, lat = window_lonlat(grib, *slices[fid])
                    groups[fid].create_array("lon", shape=lon.shape, dtype="float32",
                                             overwrite=True)[:] = lon
                    groups[fid].create_array("lat", shape=lat.shape, dtype="float32",
                                             overwrite=True)[:] = lat
            arr = extract_window(grib, *slices[fid])
            g = groups[fid]
            if "data" not in g or g["data"].shape[0] != len(per_fire[fid]):
                g.create_array("data", shape=(len(per_fire[fid]), *arr.shape), dtype="float32",
                               chunks=(1, len(CHANNELS), *arr.shape[1:]), overwrite=True)
            g["data"][pos[fid][stamp]] = arr
            g["filled"][pos[fid][stamp]] = True
        ok += 1

        if not keep_grib:
            grib.unlink(missing_ok=True)
        if n % 200 == 0 or n == len(todo):
            print(f"  {n:>6}/{len(todo)} hours  {bytes_total/1e9:6.2f} GB fetched", flush=True)

    # Archive holes: pull the +/-1 h neighbours so gaps interpolate across 2 h, same rule
    # as the per-fire path. Re-planned over the affected fires only.
    repaired = 0
    if missing_hours and repair_rounds > 0:
        affected = sorted({f for s in missing_hours for f in by_hour[s]})
        print(f"  {len(missing_hours)} hours absent from the archive, "
              f"repairing {len(affected)} affected fires")
        for fid in affected:
            r = download_event(fid, cfg, overwrite=False, repair_rounds=repair_rounds)
            repaired += len(r.get("repaired", []))

    for fid, g in groups.items():
        filled = np.asarray(g["filled"])
        g.attrs["missing"] = [s for s, f in zip(per_fire[fid], filled) if not f]

    stored = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e9
    return {"fires": len(per_fire), "skipped": len(skipped), "fire_hours": fire_hours,
            "unique_hours": unique, "dedupe": round(fire_hours / max(unique, 1), 2),
            "hours_ok": ok, "missing": len(missing_hours), "repaired": repaired,
            "gb": round(bytes_total / 1e9, 2), "stored_gb": round(stored, 2)}


def pick_mvp_fires(events_csv: Path, n: int = 5) -> pd.DataFrame:
    """Five events spanning years, regions, sizes and splits — deterministic."""
    ev = pd.read_csv(events_csv, parse_dates=["start", "end"])
    ev = ev[ev["keep"] & ~ev["touches_no_data"]]
    chosen = [
        "2017_2405",  # median-sized 2017 Sierra fire, so the set is not all megafires
        "2018_4037",  # Camp — extreme wind-driven run, tests the Phase 8 wind-shift mode
        "2020_3779",  # Creek — megafire, plume-dominated, many tiles
        "2021_3526",  # Caldor — long duration, crossed the Sierra crest
        "2022_3298",  # Mosquito — test-split year
    ]
    # The fifth fire used to be picked as the median-sized 2017 event at call time.
    # That is not deterministic across config changes: `n_timesteps` depends on
    # `labels.window_hours`, so the 6 h -> 24 h switch moved the median off 2017_2405
    # and silently orphaned its already-downloaded LANDFIRE/DEM tiles. Pinned instead.
    out = ev[ev["fire_id"].isin(chosen)].copy()
    missing = set(chosen) - set(out["fire_id"])
    if missing:
        raise SystemExit(f"MVP fires absent from {events_csv.name}: {sorted(missing)}")
    return out.sort_values("start").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download HRRR fields for fire events.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--events", default="data/processed/fire_events.csv")
    p.add_argument("--fire-id", help="single fire to fetch")
    p.add_argument("--all-mvp", action="store_true", help="fetch the 5 MVP events")
    p.add_argument("--all", action="store_true", help="every kept fire, deduped by hour")
    p.add_argument("--list-fires", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="report hours needed, fetch nothing")
    p.add_argument("--limit", type=int, help="stop after N hours per fire (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.all:
        ev = pd.read_csv(ROOT / args.events)
        ids = sorted(ev.loc[ev["keep"], "fire_id"])
        r = download_all(ids, cfg, args.overwrite, args.dry_run)
        if args.dry_run:
            print(f"\n{r['fires']} fires ({r['skipped']} skipped: pre-HRRR or no usable windows)")
            print(f"{r['fire_hours']:,} fire-hours -> {r['unique_hours']:,} unique calendar "
                  f"hours ({r['dedupe']}x dedupe)")
            print(f"estimated transfer ~{r['est_gb']} GB")
        else:
            print(f"\n{r['hours_ok']:,}/{r['unique_hours']:,} unique hours over {r['fires']} "
                  f"fires ({r['dedupe']}x dedupe)")
            print(f"{r['missing']} absent from archive, {r['repaired']} repair hours added")
            print(f"{r['gb']} GB fetched -> {r['stored_gb']} GB stored")
        return

    fires = pick_mvp_fires(ROOT / args.events)

    if args.list_fires:
        print(fires[["fire_id", "start", "lat", "lon", "n_detections", "n_timesteps", "year"]]
              .to_string(index=False))
        return

    if args.fire_id:
        fires = fires[fires["fire_id"] == args.fire_id]
        if fires.empty:
            raise SystemExit(f"{args.fire_id} is not one of the MVP fires")
    elif not args.all_mvp:
        raise SystemExit("pass --fire-id, --all-mvp, or --list-fires")

    results = []
    for _, f in fires.iterrows():
        print(f"\n{f['fire_id']}  start {f['start']}  ({f['lat']:.2f}, {f['lon']:.2f})")
        results.append(download_event(f["fire_id"], cfg, args.overwrite, args.dry_run, args.limit))

    print("\n=== summary ===")
    for r in results:
        if "error" in r:
            print(f"  {r['fire_id']}: {r['error']}")
        elif args.dry_run:
            print(f"  {r['fire_id']}: {r['hours_requested']:>5} hours  {r['span']}")
        else:
            miss = f", {len(r['missing'])} MISSING" if r["missing"] else ""
            rep = f", +{len(r['repaired'])} repair" if r.get("repaired") else ""
            print(f"  {r['fire_id']}: {r['hours_ok']}/{r['hours_requested']} hours, "
                  f"{r['mb']} MB fetched -> {r.get('stored_mb', 0)} MB stored{rep}{miss}")
    if args.dry_run:
        total = sum(r.get("hours_requested", 0) for r in results)
        print(f"  total {total:,} hours, ~{total * 5.1 / 1000:.1f} GB to fetch")
    else:
        print(f"  total {sum(r.get('mb', 0) for r in results):.1f} MB fetched, "
              f"{sum(r.get('stored_mb', 0) for r in results):.1f} MB stored")


if __name__ == "__main__":
    main()
