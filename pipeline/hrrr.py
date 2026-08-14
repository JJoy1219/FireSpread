"""HRRR weather ingestion (Phase 1.2).

HRRR surface analysis files on `s3://noaa-hrrr-bdp-pds/` are ~130 MB each, but the
four fields we need are four GRIB2 messages totalling a few MB. Every file has a
sibling `.idx` listing each message's byte offset, so we fetch only the ranges we
want with HTTP Range requests. Concatenated GRIB2 messages are themselves a valid
GRIB2 file, so the subset opens directly in cfgrib.

    hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcf00.grib2

Usage:
    python -m pipeline.hrrr --list-fires          # choose the MVP events
    python -m pipeline.hrrr --fire-id 2018_4037
    python -m pipeline.hrrr --all-mvp
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from pipeline.download import ROOT, load_config

BUCKET = "noaa-hrrr-bdp-pds"
# (short name in the .idx, level string in the .idx) -> our channel name
FIELDS = {
    ("UGRD", "10 m above ground"): "u10",
    ("VGRD", "10 m above ground"): "v10",
    ("TMP", "2 m above ground"): "t2m",
    ("RH", "2 m above ground"): "rh2m",
}
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


def download_event(
    fire_id: str,
    start: datetime,
    out_root: Path,
    lead_hours: int = 12,
    max_hours: int = 24,
    overwrite: bool = False,
) -> dict:
    """Hourly HRRR from T-lead to T+max_hours for one fire.

    Defaults follow CLAUDE.md Phase 1.2 (T-12h to T+24h), which is the right window
    for the MVP feature-stack test. Full dataset construction needs the whole active
    period instead — these fires burn for weeks, not 24 hours.
    """
    if start < HRRR_EPOCH:
        return {"fire_id": fire_id, "error": f"before HRRR epoch {HRRR_EPOCH:%Y-%m-%d}"}

    t0 = start.replace(minute=0, second=0, microsecond=0) - timedelta(hours=lead_hours)
    hours = [t0 + timedelta(hours=i) for i in range(lead_hours + max_hours + 1)]
    out_dir = out_root / fire_id

    ok = bytes_total = 0
    missing = []
    for i, dt in enumerate(hours, 1):
        res = download_hour(dt, out_dir / f"{dt:%Y%m%d_%H}z.grib2", overwrite)
        if res is None:
            missing.append(dt)
            continue
        ok += 1
        bytes_total += res[1]
        if i % 12 == 0 or i == len(hours):
            print(f"  {fire_id}  {i:>3}/{len(hours)} hours  {bytes_total/1e6:6.1f} MB")

    return {
        "fire_id": fire_id,
        "hours_requested": len(hours),
        "hours_ok": ok,
        "missing": missing,
        "mb": round(bytes_total / 1e6, 1),
        "dir": str(out_dir),
    }


def pick_mvp_fires(events_csv: Path, n: int = 5) -> pd.DataFrame:
    """Five events spanning years, regions, sizes and splits — deterministic."""
    ev = pd.read_csv(events_csv, parse_dates=["start", "end"])
    ev = ev[ev["keep"] & ~ev["touches_no_data"]]
    chosen = [
        "2018_4037",  # Camp — extreme wind-driven run, tests the Phase 8 wind-shift mode
        "2020_3779",  # Creek — megafire, plume-dominated, many tiles
        "2021_3526",  # Caldor — long duration, crossed the Sierra crest
        "2022_3298",  # Mosquito — test-split year
    ]
    out = ev[ev["fire_id"].isin(chosen)].copy()
    # Fifth: a median-sized 2017 fire, so the set is not all megafires.
    pool = ev[(ev["year"] == 2017) & (ev["n_timesteps"] >= 12)].copy()
    med = pool["n_detections"].median()
    out = pd.concat([out, pool.iloc[(pool["n_detections"] - med).abs().argsort()[:1]]])
    return out.sort_values("start").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download HRRR fields for fire events.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--events", default="data/processed/fire_events.csv")
    p.add_argument("--fire-id", help="single fire to fetch")
    p.add_argument("--all-mvp", action="store_true", help="fetch the 5 MVP events")
    p.add_argument("--list-fires", action="store_true")
    p.add_argument("--lead-hours", type=int, default=12)
    p.add_argument("--max-hours", type=int, default=24)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = ROOT / cfg["paths"]["raw_hrrr"]
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
        results.append(
            download_event(
                f["fire_id"], f["start"].to_pydatetime(), out_root,
                args.lead_hours, args.max_hours, args.overwrite,
            )
        )

    print("\n=== summary ===")
    for r in results:
        if "error" in r:
            print(f"  {r['fire_id']}: {r['error']}")
        else:
            miss = f", {len(r['missing'])} MISSING" if r["missing"] else ""
            print(f"  {r['fire_id']}: {r['hours_ok']}/{r['hours_requested']} hours, {r['mb']} MB{miss}")
    print(f"  total {sum(r.get('mb', 0) for r in results):.1f} MB")


if __name__ == "__main__":
    main()
