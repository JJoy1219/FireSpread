"""Does a stored hour actually contain that hour's weather?

`verify_hrrr.py` proves an hour is present and filled. It cannot prove the row holds
the RIGHT hour. The batched fetch indexed `data` by needed-set position while
`times`/`filled` were union-indexed, so a write could land under another hour's
timestamp -- present, filled, and wrong.

This re-downloads a sample of hours and compares them against what the store holds
for the same timestamp. A mismatch means that fire's rows are displaced. Read-only
with respect to the store: nothing here writes to the zarr.
"""
from __future__ import annotations

import argparse

import numpy as np
import zarr

from pipeline.download import ROOT, load_config
from pipeline.hrrr import (_parse_stamp, download_hour, extract_window, fire_slice,
                           read_full_field)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--fires", type=int, default=10, help="0 = every fire")
    ap.add_argument("--hours-per-fire", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/hrrr_spotcheck.csv")
    a = ap.parse_args()

    cfg = load_config(a.config)
    store = ROOT / cfg["paths"]["hrrr_windows"]
    tmp = ROOT / cfg["paths"]["raw_hrrr"] / "_spotcheck"
    tmp.mkdir(parents=True, exist_ok=True)
    margin = float(cfg["storage"]["hrrr_window_margin_cells"]) * 3.0

    rng = np.random.default_rng(a.seed)
    fires = sorted(p.stem for p in store.glob("*.zarr"))
    pick = (range(len(fires)) if a.fires == 0
            else rng.choice(len(fires), size=min(a.fires, len(fires)), replace=False))

    rows = []
    n_ok = n_bad = n_skip = 0
    print(f"{'fire':12} {'hour':14} {'max|stored-fresh|':>18}  verdict")
    for i in pick:
        fid = fires[int(i)]
        g = zarr.open_group(store / f"{fid}.zarr", mode="r")
        times = list(g.attrs.get("times", []))
        filled = np.asarray(g["filled"]) if "filled" in g else np.zeros(0, bool)
        if "data" not in g or len(times) != len(filled) or g["data"].shape[0] != len(times):
            rows.append({"fire_id": fid, "stamp": "", "delta": None, "verdict": "misaligned"})
            n_skip += 1
            continue
        avail = [j for j in range(len(times)) if filled[j]]
        if not avail:
            n_skip += 1
            continue
        for j in rng.choice(avail, size=min(a.hours_per_fire, len(avail)), replace=False):
            stamp = times[int(j)]
            grib = tmp / f"{stamp}.grib2"
            res = download_hour(_parse_stamp(stamp).to_pydatetime(), grib, overwrite=True)
            if res is None:
                n_skip += 1
                continue
            try:
                sy, sx = fire_slice(g, grib, tuple(g.attrs["bounds"]))
                fresh = extract_window(grib, sy, sx, field=read_full_field(grib))
                stored = np.asarray(g["data"][int(j)])
                d = float(np.nanmax(np.abs(stored - fresh)))
                ok = d < 1e-3
                rows.append({"fire_id": fid, "stamp": stamp, "delta": d,
                             "verdict": "ok" if ok else "MISMATCH"})
                if not ok:
                    print(f"  MISMATCH {fid} {stamp} delta {d:.4f}", flush=True)
                n_ok += ok
                n_bad += (not ok)
            finally:
                grib.unlink(missing_ok=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(ROOT / a.out, index=False)
    print(f"wrote {ROOT / a.out}")
    print(f"\nmatched {n_ok}, MISMATCHED {n_bad}, skipped {n_skip}")
    if n_bad:
        print("Mismatches mean stored rows are displaced: rebuild those fires with --overwrite.")
    elif n_ok:
        print("Sampled rows hold the correct hour.")


if __name__ == "__main__":
    main()
