"""Fire event segmentation: cluster FIRMS point detections into discrete fires.

FIRMS gives loose points, not fire IDs. We group them with a spatiotemporal
single-link clustering: two detections belong to the same event if they are
within `link_distance_km` of each other AND within `link_time_hours` in time.
Transitive closure over those links gives one component per fire, which handles
a perimeter creeping across the landscape over days without gluing together
fires that merely happen to be near each other in space or in time.

Distances are computed in EPSG:5070 metres, the same CRS the rasters use, so
clustering and rasterization agree.

Usage:
    python -m pipeline.events
    python -m pipeline.events --out data/processed/fire_events.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from pipeline.download import ROOT, load_config, load_detections


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def project(df: pd.DataFrame, crs: str = "EPSG:5070") -> pd.DataFrame:
    """Add x/y columns in the target CRS (metres)."""
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    out = df.copy()
    out["x"], out["y"] = x, y
    return out


def segment_events(
    df: pd.DataFrame,
    link_distance_km: float = 2.0,
    link_time_hours: float = 96.0,
    crs: str = "EPSG:5070",
) -> pd.DataFrame:
    """Assign an integer `event` label to each detection.

    Time-blocked sweep: detections are bucketed by day; each day's points are
    linked against every point in the preceding `link_time_hours` window. That
    keeps the KD-tree small while still catching every within-window pair.
    """
    df = project(df, crs).sort_values("acq_datetime").reset_index(drop=True)
    if df.empty:
        df["event"] = pd.Series(dtype=int)
        return df

    radius_m = link_distance_km * 1000.0
    t = df["acq_datetime"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    window = np.timedelta64(int(link_time_hours * 3600), "s")

    uf = UnionFind(len(df))
    day = df["acq_datetime"].dt.floor("D")
    # Slice boundaries per day, in the already time-sorted frame.
    day_starts = day.drop_duplicates().index.to_numpy()
    day_bounds = np.append(day_starts, len(df))

    for k in range(len(day_starts)):
        lo, hi = day_bounds[k], day_bounds[k + 1]
        # Everything still inside the time window when this day's first point fired.
        back = int(np.searchsorted(t, t[lo] - window, side="left"))
        if back == lo and hi - lo <= 1:
            continue

        tree = cKDTree(xy[back:hi])
        neighbors = tree.query_ball_point(xy[lo:hi], r=radius_m)
        for offset, nbrs in enumerate(neighbors):
            i = lo + offset
            for j_local in nbrs:
                j = back + j_local
                if j >= i:
                    continue  # each pair is handled once, from the later point
                if abs(t[i] - t[j]) <= window:
                    uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(len(df))])
    # Relabel to dense 0..N-1 ordered by first detection time.
    _, event = np.unique(roots, return_inverse=True)
    df["event"] = event
    return df


def summarize_events(
    df: pd.DataFrame,
    window_hours: int = 6,
    min_detections: int = 25,
    min_timesteps: int = 3,
    no_data_windows: list[list[str]] | None = None,
) -> pd.DataFrame:
    """One row per event, with the stats needed to filter and to name fire IDs."""
    bin_h = f"{window_hours}h"
    g = df.groupby("event")
    summary = pd.DataFrame(
        {
            "n_detections": g.size(),
            "start": g["acq_datetime"].min(),
            "end": g["acq_datetime"].max(),
            "lat": g["latitude"].mean(),
            "lon": g["longitude"].mean(),
            "x_min": g["x"].min(),
            "x_max": g["x"].max(),
            "y_min": g["y"].min(),
            "y_max": g["y"].max(),
            "frp_max": g["frp"].max(),
        }
    )
    summary["n_timesteps"] = (
        df.assign(bin=df["acq_datetime"].dt.floor(bin_h))
        .groupby("event")["bin"]
        .nunique()
    )
    summary["duration_h"] = (summary["end"] - summary["start"]).dt.total_seconds() / 3600
    summary["year"] = summary["start"].dt.year
    # Lifetime bounding-box extent in 100 m pixels. NOTE: this says whether the fire's
    # whole footprint fits one static patch — it is NOT the sample-generation gate.
    # That gate is per 6 h window (and per tile), and lives in Phase 3 features.py.
    summary["extent_px_x"] = ((summary["x_max"] - summary["x_min"]) / 100).round().astype(int)
    summary["extent_px_y"] = ((summary["y_max"] - summary["y_min"]) / 100).round().astype(int)
    summary["fits_one_patch"] = (summary["extent_px_x"] <= 256) & (summary["extent_px_y"] <= 256)

    summary["keep"] = (summary["n_detections"] >= min_detections) & (
        summary["n_timesteps"] >= min_timesteps
    )

    if no_data_windows:
        # A fire abutting a sensor outage is still a usable fire, but the windows
        # touching the outage are not usable samples: "no detections" there means
        # "no data", not "no fire". Flag it here so Phase 2 can drop those windows
        # rather than teaching the model that the fire stopped spreading.
        touches = pd.Series(False, index=summary.index)
        for w_start, w_end in no_data_windows:
            gap_start = pd.Timestamp(w_start, tz="UTC")
            gap_end = pd.Timestamp(w_end, tz="UTC") + pd.Timedelta(days=1)
            touches |= (summary["start"] <= gap_end) & (summary["end"] >= gap_start)
        summary["touches_no_data"] = touches
    else:
        summary["touches_no_data"] = False

    summary = summary.sort_values("start").reset_index()
    seq = summary.groupby("year").cumcount() + 1
    summary["fire_id"] = summary["year"].astype(str) + "_" + seq.astype(str).str.zfill(4)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Segment FIRMS detections into fire events.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--out", default="data/processed/fire_events.csv")
    p.add_argument("--detections-out", default="data/processed/detections_labeled.parquet")
    args = p.parse_args()

    cfg = load_config(args.config)
    ec, fc = cfg["events"], cfg["firms"]

    df = load_detections(
        ROOT / cfg["paths"]["raw_firms"], fc["min_confidence"], fc["vegetation_fires_only"]
    )
    print(f"{len(df):,} detections loaded")

    df = segment_events(
        df,
        link_distance_km=ec["link_distance_km"],
        link_time_hours=ec["link_time_hours"],
        crs=cfg["region"]["crs"],
    )
    summary = summarize_events(
        df,
        window_hours=cfg["labels"]["window_hours"],
        min_detections=ec["min_detections"],
        min_timesteps=ec["min_timesteps"],
        no_data_windows=fc.get("no_data_windows"),
    )

    kept = summary[summary["keep"]]
    print(f"{len(summary):,} raw clusters -> {len(kept):,} usable fire events")
    print(
        f"  of those, {int(kept['fits_one_patch'].sum()):,} fit their whole lifetime footprint "
        "in a single 256x256 100 m patch"
    )
    print("\nevents per year:")
    print(kept.groupby("year").size().to_string())
    print("\n10 largest by detection count:")
    print(
        kept.nlargest(10, "n_detections")[
            ["fire_id", "start", "lat", "lon", "n_detections", "n_timesteps", "duration_h"]
        ].to_string(index=False)
    )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    id_map = summary.set_index("event")["fire_id"]
    df["fire_id"] = df["event"].map(id_map)
    det_out = ROOT / args.detections_out
    df[df["event"].isin(kept["event"])].to_parquet(det_out, index=False)
    print(f"\nwrote {out}\nwrote {det_out}")


if __name__ == "__main__":
    main()
