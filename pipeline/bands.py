"""Distance bands from the active perimeter, and the per-band class weights.

Why this exists
---------------
Measured on the test split, model skill is almost entirely a near-ring effect:
CSI 0.30 at 0.2 km, 0.002 by 2 km, 0.000 past 3.2 km, while 12% of all real
growth lies beyond 1.2 km. A single global `pos_weight` is a large part of the
cause. Positives are ~14.6% of pixels one cell out but ~0.18% at 3 km, so the
balancing weight each band needs, `(1-p)/p`, ranges from about 6 to about 550.
Applying one value (103) over-weights near positives ~17x and under-weights far
ones ~5x, which tells the model to pile probability mass on the perimeter.

`band_index` maps euclidean distance to the nearest burned cell onto these bands,
and `band_pos_weights` turns train-split base rates into per-band weights.

Distances are in cells; the grid is 100 m, so 1 cell = 100 m.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Upper edges in cells. Chosen to be roughly log-spaced, because base rate falls
# off geometrically, and to keep every band wide enough to hold enough positives
# for a stable rate estimate.
EDGES: tuple[float, ...] = (0, 1, 2, 3, 5, 8, 12, 20, 32, np.inf)
N_BANDS = len(EDGES) - 1

# First band counted as "far". Band 6 starts at 1.2 km, which is where the baseline
# model's POD collapses to 0.002 while 12% of all real growth still lies beyond it.
FAR_BAND_MIN = 6

# Labels for reporting, in km.
BAND_LABELS = tuple(
    f"{EDGES[b]*0.1:.1f}-{EDGES[b+1]*0.1:.1f} km" if np.isfinite(EDGES[b + 1])
    else f"{EDGES[b]*0.1:.1f}+ km"
    for b in range(N_BANDS)
)


def distance_to_burn(cur: np.ndarray) -> np.ndarray:
    """Euclidean distance in cells from every pixel to the nearest burned cell."""
    from scipy.ndimage import distance_transform_edt
    if not cur.any():                      # no fire in tile: everything is "far"
        return np.full(cur.shape, np.inf, dtype="float32")
    return distance_transform_edt(~cur).astype("float32")


def band_index(dist: np.ndarray) -> np.ndarray:
    """Band index per pixel, as int8. `np.digitize` with the open last edge."""
    return (np.digitize(dist, EDGES[1:-1], right=False)).astype("int8")


def band_pos_weights(stats: dict, cap: float, floor: float = 1.0) -> np.ndarray:
    """Per-band positive weight `(1-p)/p` from train base rates, capped.

    The cap is not cosmetic. The outermost band's base rate implies a weight in
    the thousands, and far-field positives are disproportionately spotting,
    separate ignitions and fire-merge artifacts -- the noisiest labels in the
    dataset, and flagged as a labelling ambiguity rather than a model failure in
    CLAUDE.md Phase 8. Uncapped, the loss would chase that noise hardest.
    """
    w = np.ones(N_BANDS, dtype="float32")
    for b in range(N_BANDS):
        # Band 0 is empty by construction: only burned cells sit at distance 0, and a
        # burned cell can never be a target. An empty band gets weight 1, never the cap,
        # so a band that goes empty for another reason cannot silently dominate the loss.
        if int(stats["n_pixels"][b]) == 0:
            continue
        p = float(stats["base_rate"][b])
        w[b] = cap if p <= 0 else float(np.clip((1.0 - p) / p, floor, cap))
    return w


def load_band_stats(cfg: dict) -> dict:
    p = ROOT / cfg["paths"].get("band_stats", "configs/band_stats.json")
    if not p.exists():
        raise SystemExit(
            f"{p} missing -- build it with:\n"
            f"  python -m analysis.band_stats --config <cfg>")
    s = json.loads(p.read_text())
    if len(s["base_rate"]) != N_BANDS:
        raise SystemExit(f"{p} has {len(s['base_rate'])} bands, code expects {N_BANDS}; "
                         "EDGES changed since it was built, so rebuild it.")
    if s.get("split") != "train":
        raise SystemExit(f"{p} was built on split {s.get('split')!r}; weights must come "
                         "from train only or val/test leak into the loss.")
    return s
