"""Derived channels that hand the model a relationship it cannot easily infer.

Motivation. Occlusion says the model draws 81-93% of its skill from the burn mask
and essentially nothing from wind, at every threshold, at 24 h and at 12 h. But the
labels DO respond to wind: over 469 training samples the angle between the growth
vector and the tile-mean wind vector gives mean cos +0.166, and it rises with wind
speed (+0.027 / +0.157 / +0.309 / +0.171 by quartile). That monotone rise is what
rules out a sign error or a time misalignment, so the signal is real, weak, and
unused.

The plausible reason is that the signal is hard to READ in the form supplied. A
256 px tile at 100 m spans 25.6 km against HRRR's 3 km grid, so wind is close to
constant across it. To use it the network must combine a near-uniform vector field
with the perimeter geometry and work out, per pixel, "am I downwind of the nearest
burning cell?" That is a nonlocal computation over two channels, and convolutions
are poorly suited to it.

`downwind_field` computes exactly that quantity directly:

    for each pixel, take the unit vector from its NEAREST burning cell to itself,
    and project the local wind vector onto it.

Positive means the pixel lies downwind of the fire, negative upwind, and the
magnitude carries wind speed, which matters because the observed relationship
strengthens with it. Units are m/s.

Flip augmentation. This is a scalar, not a vector. Under a mirror the outward unit
vector and the wind vector both negate in the mirrored axis, so their dot product is
unchanged. The field therefore moves with the pixels but must NOT be sign-flipped
the way u10/v10 are, and it must stay out of DIRECTION_GROUPS.
"""
from __future__ import annotations

import numpy as np


def downwind_field(cur: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Wind projected onto the outward direction from the nearest burning cell.

    `cur` is the boolean fire extent, `u` eastward and `v` northward wind in m/s.
    Returns a float32 field in m/s: positive downwind, negative upwind.
    """
    from scipy.ndimage import distance_transform_edt

    if not cur.any():
        # No fire in the tile, so "outward from the fire" is undefined. Zero is the
        # right filler: it is what an exactly crosswind pixel scores.
        return np.zeros(cur.shape, dtype="float32")

    # Indices of the nearest burning cell for every pixel. The transform works on the
    # complement, so its "nearest feature" is the nearest True in `cur`.
    _, idx = distance_transform_edt(~cur, return_indices=True)
    rows, cols = np.indices(cur.shape)
    d_row = rows - idx[0]          # +row is southward on a north-up grid
    d_col = cols - idx[1]          # +col is eastward

    east = d_col.astype("float32")
    north = -d_row.astype("float32")
    norm = np.hypot(east, north)
    # Burning cells are their own nearest feature, giving a zero-length vector.
    safe = norm > 0
    east = np.divide(east, norm, out=np.zeros_like(east), where=safe)
    north = np.divide(north, norm, out=np.zeros_like(north), where=safe)

    return (east * u.astype("float32") + north * v.astype("float32")).astype("float32")


def downwind_stack(burn: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """`downwind_field` for each sequence step. Shapes `(T, H, W)` in, `(T, H, W)` out."""
    return np.stack([downwind_field(burn[k] > 0.5, u[k], v[k])
                     for k in range(burn.shape[0])]).astype("float32")
