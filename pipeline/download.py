"""Data fetching. Currently: FIRMS VIIRS active-fire detections (Phase 1.1).

The FIRMS area API serves at most 5 days per request:

    https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{bbox}/{day_range}/{start_date}

where bbox is west,south,east,north. Anything above 5 returns HTTP 400
("Invalid day range. Expects [1..5]") despite the docs suggesting 10.
Requests are rate limited (5000 per 10 minutes per key), comfortably above
what a 9-year pull needs (~660).

Usage:
    set FIRMS_MAP_KEY=...        (or put it in .env)
    python -m pipeline.download firms
    python -m pipeline.download firms --start 2020-01-01 --end 2020-12-31
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAY_RANGE = 5  # hard limit imposed by the FIRMS area API (verified empirically)


def load_config(path: str | Path = "configs/baseline.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_map_key() -> str:
    """Read the FIRMS MAP_KEY from the environment or a local .env file."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if key:
        return key.strip()

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("FIRMS_MAP_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")

    sys.exit(
        "FIRMS_MAP_KEY is not set.\n"
        "  1. Request a key at https://firms.modaps.eosdis.nasa.gov/api/area/\n"
        "  2. Then either:  set FIRMS_MAP_KEY=your_key\n"
        "     or write it into a .env file at the project root as FIRMS_MAP_KEY=your_key"
    )


def check_key(map_key: str) -> None:
    """Verify the key works and report remaining transaction quota."""
    url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={map_key}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        print(f"MAP_KEY status: {r.text.strip()[:400]}")
    except requests.RequestException as exc:
        print(f"warning: could not verify MAP_KEY ({exc}); continuing anyway")


def _chunks(start: date, end: date, size: int = MAX_DAY_RANGE, anchor: date | None = None):
    """Yield (chunk_start, n_days) covering [start, end] inclusive.

    Chunk boundaries are snapped to a fixed `anchor` grid so that a narrow run
    (`--start 2020-08-01`) produces the same filenames as the same span inside a
    full-archive run. Without this, partial runs land off-grid and leave
    overlapping duplicate files behind.
    """
    cur = start
    if anchor is not None and start > anchor:
        cur = anchor + timedelta(days=((start - anchor).days // size) * size)
    while cur <= end:
        n = min(size, (end - cur).days + 1)
        yield cur, n
        cur += timedelta(days=n)


def _fetch_chunk(url: str, attempts: int = 3) -> str | None:
    """GET one chunk, retrying transient failures. Returns CSV text, or None."""
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            text = r.text
            if r.status_code == 200 and text.lstrip().lower().startswith(
                ("country_id", "latitude")
            ):
                return text
            # FIRMS reports problems as a plain-text body — surface it verbatim,
            # since it names the actual constraint (bad day range, bad source, ...).
            reason = f"HTTP {r.status_code}: {text.strip()[:200]!r}"
            if r.status_code == 400:
                print(f"    {reason}")
                return None  # a malformed request will not fix itself on retry

        if attempt < attempts:
            backoff = 5 * attempt
            print(f"    attempt {attempt}/{attempts} failed ({reason}); retrying in {backoff}s")
            time.sleep(backoff)
        else:
            print(f"    giving up after {attempts} attempts ({reason})")
    return None


def download_firms(
    out_dir: Path,
    bbox: list[float],
    source: str,
    start: date,
    end: date,
    map_key: str,
    overwrite: bool = False,
    pause_s: float = 0.5,
    anchor: date | None = None,
) -> tuple[list[Path], list[date]]:
    """Download FIRMS detections in <=5-day chunks.

    Resumable: chunks already on disk are skipped. Returns (files, failed_chunk_starts)
    so a long run cannot silently lose days.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # API wants west,south,east,north — same order as the config bbox.
    bbox_str = ",".join(str(v) for v in bbox)

    written: list[Path] = []
    failed: list[date] = []
    chunks = list(_chunks(start, end, anchor=anchor))
    for i, (chunk_start, n_days) in enumerate(chunks, 1):
        dest = out_dir / f"{source}_{chunk_start.isoformat()}_{n_days}d.csv"
        if dest.exists() and not overwrite:
            written.append(dest)
            continue

        url = f"{API}/{map_key}/{source}/{bbox_str}/{n_days}/{chunk_start.isoformat()}"
        text = _fetch_chunk(url)
        if text is None:
            print(f"[{i}/{len(chunks)}] {chunk_start} FAILED")
            failed.append(chunk_start)
            continue

        dest.write_text(text, encoding="utf-8")
        n_rows = max(text.count("\n") - 1, 0)
        written.append(dest)
        # A header-only CSV is a valid HTTP 200 but is almost always a sensor
        # outage rather than a genuinely fire-free window — see the 2022 S-NPP
        # gap documented in the README. Call it out instead of banking it silently.
        note = "   <-- EMPTY, verify against another sensor" if n_rows == 0 else ""
        print(f"[{i}/{len(chunks)}] {chunk_start} +{n_days}d -> {n_rows:>6} detections{note}")
        time.sleep(pause_s)

    return written, failed


def audit_archive(firms_dir: Path, start: date, end: date) -> None:
    """Report empty chunks and multi-day detection gaps in what has been downloaded."""
    empty = []
    for f in sorted(firms_dir.glob("*.csv")):
        with open(f, encoding="utf-8") as fh:
            if sum(1 for _ in fh) <= 1:
                empty.append(f.name)
    print(f"empty chunk files: {len(empty)}")
    for name in empty:
        print(f"  {name}")

    det = load_detections(firms_dir)
    have = set(pd.to_datetime(det["acq_datetime"]).dt.date.unique())
    missing = sorted(set(pd.date_range(start, end, freq="D").date) - have)

    runs: list[list] = []
    for d in missing:
        if runs and (d - runs[-1][-1]).days == 1:
            runs[-1].append(d)
        else:
            runs.append([d])

    print(f"\n{len(missing)} of {(end - start).days + 1} days have no detections")
    print("runs of >= 3 consecutive empty days:")
    for r in runs:
        if len(r) < 3:
            continue
        season = " <-- FIRE SEASON, likely a sensor outage" if r[0].month in (6, 7, 8, 9, 10) else ""
        print(f"  {r[0]} .. {r[-1]}  ({len(r)} days){season}")


def load_detections(
    firms_dir: Path,
    min_confidence: str = "nominal",
    vegetation_only: bool = True,
) -> pd.DataFrame:
    """Concatenate downloaded CSVs into one clean, filtered, time-sorted frame.

    Returns columns: latitude, longitude, acq_datetime (UTC), confidence, frp, satellite.
    """
    files = sorted(firms_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no FIRMS CSVs in {firms_dir}; run `download.py firms` first")

    # Header-only chunks (sensor outages) would otherwise poison the concat dtypes.
    frames = [d for d in (pd.read_csv(f) for f in files) if not d.empty]
    df = pd.concat(frames, ignore_index=True)

    # VIIRS confidence is categorical: l(ow) / n(ominal) / h(igh).
    rank = {"l": 0, "n": 1, "h": 2}
    want = {"low": 0, "nominal": 1, "high": 2}[min_confidence]
    conf = df["confidence"].astype(str).str.strip().str.lower().str[0]
    df = df[conf.map(rank).fillna(-1) >= want]

    # type: 0=vegetation fire, 1=active volcano, 2=other static land, 3=offshore.
    # Present in SP (archive) products; absent from some NRT exports.
    if vegetation_only and "type" in df.columns:
        df = df[df["type"] == 0]

    # acq_time is HHMM as an int (e.g. 934 -> 09:34), UTC.
    acq_time = df["acq_time"].astype(int).astype(str).str.zfill(4)
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + acq_time, format="%Y-%m-%d%H%M", utc=True
    )

    keep = ["latitude", "longitude", "acq_datetime", "confidence", "frp"]
    if "satellite" in df.columns:
        keep.append("satellite")

    df = (
        df[keep]
        .drop_duplicates(subset=["latitude", "longitude", "acq_datetime"])
        .sort_values("acq_datetime")
        .reset_index(drop=True)
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Download raw datasets.")
    p.add_argument("dataset", choices=["firms", "audit"])
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--start", help="YYYY-MM-DD (overrides config)")
    p.add_argument("--end", help="YYYY-MM-DD (overrides config)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    fc, paths = cfg["firms"], cfg["paths"]

    start = date.fromisoformat(args.start or fc["start_date"])
    end = date.fromisoformat(args.end or fc["end_date"])
    if end < start:
        sys.exit("--end is before --start")

    out_dir = ROOT / paths["raw_firms"]
    if args.dataset == "audit":
        audit_archive(out_dir, start, end)
        return

    map_key = get_map_key()
    check_key(map_key)

    files, failed = download_firms(
        out_dir=out_dir,
        bbox=cfg["region"]["bbox"],
        source=fc["source"],
        start=start,
        end=end,
        map_key=map_key,
        overwrite=args.overwrite,
        anchor=date.fromisoformat(fc["start_date"]),
    )
    print(f"\n{len(files)} chunk files in {out_dir}")
    if failed:
        print(f"WARNING: {len(failed)} chunks failed and are missing from the archive:")
        for d in failed:
            print(f"  {d}")
        print("Rerun the same command to retry only the missing chunks.")

    df = load_detections(out_dir, fc["min_confidence"], fc["vegetation_fires_only"])
    print(
        f"{len(df):,} detections after filtering "
        f"({df['acq_datetime'].min()} .. {df['acq_datetime'].max()})"
    )


if __name__ == "__main__":
    main()
