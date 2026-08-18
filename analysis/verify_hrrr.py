"""Verify the HRRR store actually serves every hour the config needs.

Written after a self-inflicted regression. Topping the store up for the 12 h arm
flipped previously-filled hours to unfilled, and on at least one fire removed
entries outright. It went unnoticed because the check at the time compared the
`times` attribute before and after and found nothing lost -- but `times` is only
the index. What gates the data is the `filled` boolean and the `data` array, and
those are what `_hour_field` consults:

    i = pos.get(f"{t:%Y%m%d_%H}z")
    if i is not None and filled[i]:

So an hour can be listed and still be a hole. This checks the three arrays agree
in length and that every hour the config needs is present AND filled, which is the
property the Dataset depends on.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import zarr

from pipeline.download import ROOT, load_config
from pipeline.hrrr import needed_hours


def check_fire(fid: str, cfg: dict, store) -> dict:
    p = store / f"{fid}.zarr"
    if not p.exists():
        return {"fire_id": fid, "status": "no_store"}
    g = zarr.open_group(p, mode="r")
    times = list(g.attrs.get("times", []))
    filled = np.asarray(g["filled"]) if "filled" in g else np.zeros(0, dtype=bool)
    n_data = g["data"].shape[0] if "data" in g else 0
    aligned = len(times) == len(filled) == n_data

    try:
        need = {f"{t:%Y%m%d_%H}z" for t in needed_hours(fid, cfg)}
    except Exception as exc:
        return {"fire_id": fid, "status": f"needed_hours_failed: {str(exc)[:40]}"}

    pos = {s: i for i, s in enumerate(times)}
    absent = {h for h in need if h not in pos}
    # Present but not filled is the failure mode the old check missed.
    hollow = {h for h in need if h in pos and not bool(filled[pos[h]])}
    return {
        "fire_id": fid, "status": "ok" if (aligned and not absent and not hollow) else "BAD",
        "aligned": aligned, "n_times": len(times), "n_filled": int(filled.sum()),
        "n_data": n_data, "need": len(need), "absent": len(absent), "hollow": len(hollow),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out", default="data/processed/hrrr_verify.csv")
    a = ap.parse_args()

    cfg = load_config(a.config)
    store = ROOT / cfg["paths"]["hrrr_windows"]
    fires = sorted(p.stem for p in store.glob("*.zarr"))
    rows = [check_fire(f, cfg, store) for f in fires]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / a.out, index=False)

    bad = df[df["status"] != "ok"]
    print(f"config {a.config}")
    print(f"  fires checked      {len(df):,}")
    print(f"  healthy            {(df['status'] == 'ok').sum():,}")
    print(f"  unhealthy          {len(bad):,}")
    if "aligned" in df:
        print(f"  array misalignment {int((df['aligned'] == False).sum()):,}")
        print(f"  needed but ABSENT  {int(df['absent'].fillna(0).sum()):,} hours")
        print(f"  needed but HOLLOW  {int(df['hollow'].fillna(0).sum()):,} hours")
    print(f"wrote {ROOT / a.out}")


if __name__ == "__main__":
    main()
