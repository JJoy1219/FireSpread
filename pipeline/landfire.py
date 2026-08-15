"""LANDFIRE fuels ingestion (Phase 1.3).

Sourced from the LANDFIRE Product Service (LFPS). Note the documented ArcGIS
`GPServer/.../submitJob` endpoint is shadowed by the current website and returns HTML;
the live API is:

    GET /api/products                       full product catalog
    GET /api/job/submit?Email=&Layer_List=&Area_of_Interest=&Output_Projection=
    GET /api/job/status?JobId=              poll until Succeeded, then fetch outputFile

LFPS clips and reprojects server-side, so we ask for a per-fire AOI already in
EPSG:5070 rather than pulling the 30 m CONUS mosaic. Output is a zip holding one
multi-band GeoTIFF, bands in `Layer_List` order.

    python -m pipeline.landfire --all-mvp
    python -m pipeline.landfire --fire-id 2018_4037
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import rasterio
import requests

from pipeline.dem import bbox_for_fire
from pipeline.download import ROOT, load_config

API = "https://lfps.usgs.gov/api"


def get_email() -> str:
    email = os.environ.get("LFPS_EMAIL")
    if email:
        return email.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("LFPS_EMAIL="):
                return line.split("=", 1)[1].strip().strip("\"'")
    sys.exit("LFPS_EMAIL is not set. Add LFPS_EMAIL=you@example.com to .env")


def version_for_fire(year: int, cfg: dict) -> str:
    """Pick the fuel version, never one published after the fire (see config notes)."""
    lc = cfg["landfire"]
    if lc["version_strategy"] == "fixed":
        return lc["fixed_version"]
    if lc["version_strategy"] != "latest_prior":
        raise ValueError(f"unknown version_strategy {lc['version_strategy']!r}")
    prior = [v for v in lc["available_fuel_versions"] if int(v[2:]) < year]
    if not prior:
        # 2015-2016 fires: nothing earlier exists, fall back and let the caller flag it.
        return min(lc["available_fuel_versions"], key=lambda v: int(v[2:]))
    return max(prior, key=lambda v: int(v[2:]))


def submit(layers: list[str], aoi: tuple[float, float, float, float], email: str,
           projection: int = 5070) -> str:
    p = {
        "Email": email,
        "Layer_List": ";".join(layers),
        "Area_of_Interest": f"{aoi[0]:.5f} {aoi[1]:.5f} {aoi[2]:.5f} {aoi[3]:.5f}",
        "Output_Projection": str(projection),
    }
    r = requests.get(f"{API}/job/submit", params=p, timeout=120)
    r.raise_for_status()
    j = r.json()
    if not j.get("jobId"):
        raise RuntimeError(f"submit failed: {json.dumps(j)[:400]}")
    return j["jobId"]


def wait(job_id: str, timeout_s: int = 1800, poll_s: int = 10) -> str:
    """Block until the job finishes; return the output zip URL."""
    waited = 0
    while waited < timeout_s:
        time.sleep(poll_s)
        waited += poll_s
        s = requests.get(f"{API}/job/status", params={"JobId": job_id}, timeout=90).json()
        status = str(s.get("status", ""))
        if status.lower() == "succeeded":
            out = s.get("outputFile")
            if not out:
                raise RuntimeError(f"job {job_id} succeeded with no outputFile")
            return out
        if status.lower() in ("failed", "cancelled", "timedout"):
            msgs = [m.get("description", "") for m in s.get("messages", [])][-4:]
            raise RuntimeError(f"job {job_id} {status}: {msgs}")
        if waited % 60 == 0:
            q = s.get("queuePosition")
            print(f"    {waited}s {status}" + (f" (queue {q})" if q and q > 0 else ""))
    raise TimeoutError(f"job {job_id} still running after {timeout_s}s")


def fetch_zip(url: str, dest_tif: Path, band_names: list[str]) -> Path:
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = [n for n in z.namelist() if n.lower().endswith(".tif")]
    if not names:
        raise RuntimeError(f"no .tif in output zip ({z.namelist()})")

    dest_tif.parent.mkdir(parents=True, exist_ok=True)
    dest_tif.write_bytes(z.read(names[0]))
    # Band order follows Layer_List but nothing in the file records it — write a sidecar.
    dest_tif.with_suffix(".bands.json").write_text(
        json.dumps({"bands": band_names}, indent=2), encoding="utf-8"
    )
    return dest_tif


def fetch_for_fire(row: pd.Series, cfg: dict, email: str, overwrite: bool = False) -> dict:
    lc = cfg["landfire"]
    year = int(row["year"])
    version = version_for_fire(year, cfg)
    layers = [f"{version}_{a}" for a in lc["layers"]]
    dest = ROOT / cfg["paths"]["raw_landfire"] / f"{row['fire_id']}_{version}.tif"

    result = {
        "fire_id": row["fire_id"],
        "version": version,
        "contaminated": year < lc["contaminated_before_year"],
        "path": str(dest),
    }
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        result["status"] = "cached"
        return result

    aoi = bbox_for_fire(row, lc["aoi_margin_km"] * 1000)
    job = submit(layers, aoi, email, lc["output_projection"])
    print(f"    job {job}  layers {','.join(layers)}")
    url = wait(job)
    fetch_zip(url, dest, lc["layers"])
    result["status"] = "downloaded"
    return result


def main() -> None:
    from pipeline.hrrr import pick_mvp_fires

    p = argparse.ArgumentParser(description="Download LANDFIRE fuels per fire via LFPS.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--events", default="data/processed/fire_events.csv")
    p.add_argument("--fire-id")
    p.add_argument("--all-mvp", action="store_true")
    p.add_argument("--all", action="store_true", help="every kept fire")
    p.add_argument("--quiet", action="store_true", help="one line per fire, for long runs")
    p.add_argument("--workers", type=int, default=6, help="concurrent LFPS jobs")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    email = get_email()
    if args.all:
        ev = pd.read_csv(ROOT / args.events)
        fires = ev[ev["keep"]].sort_values("fire_id")
    else:
        fires = pick_mvp_fires(ROOT / args.events)
        if args.fire_id:
            fires = fires[fires["fire_id"] == args.fire_id]
        elif not args.all_mvp:
            raise SystemExit("pass --fire-id, --all-mvp or --all")

    print(f"strategy={cfg['landfire']['version_strategy']} "
          f"layers={cfg['landfire']['layers']}  {len(fires)} fires, "
          f"{args.workers} workers\n")

    def one(f: pd.Series) -> dict:
        r = fetch_for_fire(f, cfg, email, args.overwrite)
        with rasterio.open(r["path"]) as ds:
            r["shape"] = f"{ds.width}x{ds.height}"
            r["mb"] = round(Path(r["path"]).stat().st_size / 1e6, 1)
            r["stats"] = "; ".join(
                f"{name} {ds.read(b, masked=True).min()}-{ds.read(b, masked=True).max()}"
                for b, name in enumerate(cfg["landfire"]["layers"], 1))
        return r

    rows, failures = [], []
    # LFPS is a job queue: almost all of the ~100 s per fire is spent waiting on the
    # server, not transferring, so a handful of concurrent jobs turns a 17 h serial run
    # into a few hours. Kept deliberately modest -- this is a public service.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, f): f["fire_id"] for _, f in fires.iterrows()}
        for i, fut in enumerate(as_completed(futures), 1):
            fid = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:
                # One fire failing must not lose the other 621; LFPS is a live service
                # and transient job failures are expected over a run this long.
                print(f"  {fid} FAILED: {type(exc).__name__}: {exc}", flush=True)
                failures.append(fid)
                continue
            if args.quiet:
                if i % 25 == 0 or i == len(fires):
                    print(f"  {i:>4}/{len(fires)}  {len(failures)} failed", flush=True)
            else:
                flag = "  [FUEL MAP POSTDATES FIRE]" if r["contaminated"] else ""
                print(f"  {fid}  {r['status']}  {r['version']}  {r['shape']}  "
                      f"{r['mb']} MB | {r['stats']}{flag}", flush=True)
            rows.append(r)

    out = ROOT / "data/processed/landfire_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{len(rows)} fires ok, {len(failures)} failed")
    if failures:
        print(f"failed: {failures[:20]}{' ...' if len(failures) > 20 else ''}")
        print("rerun the same command to retry — completed fires are skipped")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
