"""PyTorch Dataset over the tiled sample index (Phase 4).

Samples are `(fire_id, t_index, row0, col0)` rows of `sample_index.parquet`. Nothing is
materialised: each `__getitem__` crops the burn mask from the label zarr, crops the static
stack, and regrids the HRRR window for that tile on demand. See the storage section of the
README — materialising tiles would cost 127-254 GB against ~35 MB of per-fire rasters here.

    python -m pipeline.dataset splits     # write train/val/test fire lists
    python -m pipeline.dataset norm       # per-channel mean/std, TRAIN fires only
    python -m pipeline.dataset check      # shapes, NaNs, flip correctness, throughput
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from pipeline.download import ROOT, load_config
from pipeline.features import WEATHER_CHANNELS, static_tile, weather_tile

# Fallback only; `sampling.split_years` in the config is authoritative. See the note
# there for why this is not the CLAUDE.md Phase 7 boundary.
DEFAULT_SPLIT_YEARS = {"train": (2015, 2020), "val": (2021, 2021), "test": (2022, 2023)}


def split_years(cfg: dict) -> dict[str, tuple[int, int]]:
    raw = cfg.get("sampling", {}).get("split_years")
    if not raw:
        return dict(DEFAULT_SPLIT_YEARS)
    years = {k: (int(v[0]), int(v[1])) for k, v in raw.items()}
    if set(years) != set(DEFAULT_SPLIT_YEARS):
        raise SystemExit(f"sampling.split_years must define {sorted(DEFAULT_SPLIT_YEARS)}")
    spans = sorted(years.values())
    for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
        if b_lo <= a_hi:
            raise SystemExit(f"split_years overlap: {a_lo}-{a_hi} and {b_lo}-{b_hi}")
    return years
# Channels that must NOT be normalised: the burn mask is already the 0/1 the loss expects,
# and rescaling it would put the model's own prediction target on a different scale from
# the input it conditions on.
NO_NORM = {"burn_mask"}

# Signed direction components, normalised with **mean forced to zero and a std shared
# across each group**. Two reasons, both correctness rather than taste:
#
#  1. Flip augmentation negates these channels *after* normalisation. With a non-zero
#     mean, negating gives (mu - x)/sigma where the true flipped value is (-x - mu)/sigma
#     -- off by 2*mu/sigma. Zero mean makes negation and normalisation commute, which is
#     precisely the symmetry the augmentation asserts.
#  2. The lag channels are one variable sampled at three times. Per-lag means would make
#     a constant wind field look like it was changing across lags, inventing a temporal
#     signal in the channels that exist to carry temporal change.
DIRECTION_GROUPS = (
    ("u10_lag0", "u10_lag6", "u10_lag12", "v10_lag0", "v10_lag6", "v10_lag12"),
    ("aspect_sin", "aspect_cos"),
)


def channel_names(cfg: dict) -> list[str]:
    """The channel list actually produced, after `elevation_mode` is applied."""
    names = list(cfg["model"]["channels"])
    if str(cfg["model"].get("elevation_mode", "fire_centred")) == "none":
        names = [c for c in names if c != "elevation_centred"]
    return names


def write_splits(cfg: dict, events_csv: str = "data/processed/fire_events.csv") -> dict:
    """Fire-level chronological splits, per CLAUDE.md Phase 7.

    Grouped by `fire_id` and never by sample: tiles of one fire overlap 50% and share
    terrain and weather, so splitting at sample level would leak badly and inflate CSI.
    """
    if cfg["sampling"]["group_splits_by"] != "fire_id":
        raise SystemExit("sampling.group_splits_by must be fire_id")
    ev = pd.read_csv(ROOT / events_csv)
    ev = ev[ev["keep"]]
    out_dir = ROOT / cfg["paths"]["splits"]
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for name, (lo, hi) in split_years(cfg).items():
        ids = sorted(ev.loc[(ev["year"] >= lo) & (ev["year"] <= hi), "fire_id"])
        (out_dir / f"{name}_fires.txt").write_text("\n".join(ids) + "\n")
        counts[name] = len(ids)
    return counts


def read_split(cfg: dict, split: str) -> list[str]:
    path = ROOT / cfg["paths"]["splits"] / f"{split}_fires.txt"
    if not path.exists():
        raise SystemExit(f"{path} missing — run `python -m pipeline.dataset splits` first")
    return [ln.strip() for ln in path.read_text().split("\n") if ln.strip()]


class WildfireDataset(Dataset):
    def __init__(self, cfg: dict, split: str, norm_stats: dict | None = None,
                 augment: bool = False, index_path: str | None = None):
        self.cfg = cfg
        self.split = split
        self.augment = augment
        self.patch = int(cfg["grid"]["patch_size"])
        self.t_steps = int(cfg["model"]["t_steps"])
        self.channels = channel_names(cfg)
        self.burn_mode = str(cfg["model"].get("burn_mask_mode", "cumulative"))
        aug = cfg.get("sampling", {}).get("augment", {}) or {}
        self.noise_std = float(aug.get("noise_std", 0.0))
        self.chan_drop = float(aug.get("channel_dropout", 0.0))

        idx = pd.read_parquet(ROOT / (index_path or cfg["paths"].get(
            "sample_index", "data/processed/sample_index.parquet")))
        fires = set(read_split(cfg, split))
        self.index = idx[idx["fire_id"].isin(fires)].reset_index(drop=True)

        if norm_stats is None:
            p = ROOT / cfg["paths"]["norm_stats"]
            norm_stats = json.loads(p.read_text()) if p.exists() else None
        if norm_stats:
            self.mean = np.array([norm_stats[c]["mean"] for c in self.channels], dtype="float32")
            self.std = np.array([norm_stats[c]["std"] for c in self.channels], dtype="float32")
        else:
            self.mean = np.zeros(len(self.channels), dtype="float32")
            self.std = np.ones(len(self.channels), dtype="float32")
        for i, c in enumerate(self.channels):          # keep the burn mask at 0/1
            if c in NO_NORM:
                self.mean[i], self.std[i] = 0.0, 1.0

        # Opened lazily so DataLoader workers each get their own handles.
        self._labels: dict[str, zarr.Group] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _label_group(self, fire_id: str):
        if fire_id not in self._labels:
            self._labels[fire_id] = zarr.open_group(
                ROOT / "data/processed/labels" / f"{fire_id}.zarr", mode="r")
        return self._labels[fire_id]

    def _burn(self, fire_id: str, t: int, row0: int, col0: int) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative burn for each sequence step, and the target for window t+1.

        Step s carries the cumulative burn at that step's own window, matching the
        weather semantics: step s corresponds to window `t - (t_steps-1-s)`.
        """
        g = self._label_group(fire_id)
        burn = g["burn_new"]
        p = self.patch
        sl = (slice(row0, row0 + p), slice(col0, col0 + p))

        cum = np.zeros((p, p), dtype=bool)
        steps = np.zeros((self.t_steps, p, p), dtype="float32")
        first = t - (self.t_steps - 1)
        for k in range(0, t + 1):                      # accumulate from ignition
            new_k = np.asarray(burn[k][sl]) > 0
            cum |= new_k
            if k >= first:
                # `cumulative` hands each step the whole history as a static channel, so
                # the feed-forward path can read "where the fire has been" without any
                # recurrence — which is one reason the ConvLSTM ablation found the
                # recurrence redundant. `incremental` gives each step only that window's
                # new burn, so history is recoverable ONLY by integrating the sequence.
                steps[k - first] = cum if self.burn_mode == "cumulative" else new_k
        target = (np.asarray(burn[t + 1][sl]) > 0) & ~cum
        return steps, target.astype("float32")

    def __getitem__(self, i: int) -> dict:
        r = self.index.iloc[i]
        fid, t = str(r["fire_id"]), int(r["t_index"])
        row0, col0 = int(r["row0"]), int(r["col0"])

        burn, target = self._burn(fid, t, row0, col0)
        wx = weather_tile(fid, t, self.cfg, row0, col0, self.patch)
        if wx is None:
            raise RuntimeError(f"{fid} t={t} has an unbridgeable weather gap; "
                               "rebuild the index or repair the HRRR store")
        stat, fuel = static_tile(fid, self.cfg, row0, col0, self.patch)

        # burn (1) + weather (9) + static (7), static repeated across the sequence.
        x = np.concatenate([
            burn[:, None],
            wx,
            np.repeat(stat[None], self.t_steps, axis=0),
        ], axis=1)
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        if self.augment:
            x, target, fuel = self._flip(x, target, fuel)
            x = self._jitter(x)

        return {
            "input": torch.from_numpy(np.ascontiguousarray(x, dtype="float32")),
            "fuel": torch.from_numpy(np.ascontiguousarray(fuel)).long(),
            "target": torch.from_numpy(np.ascontiguousarray(target, dtype="float32")),
            "meta": {"fire_id": fid, "t_index": t, "row0": row0, "col0": col0},
        }

    def _jitter(self, x: np.ndarray) -> np.ndarray:
        """Gaussian noise and channel dropout on the CONTINUOUS channels only.

        The burn mask is excluded from both. CLAUDE.md forbids intensity shifts on it, and
        the reason is substantive rather than stylistic: it is the one channel the model
        conditions on to know where the fire currently is, and it shares its 0/1 scale with
        the prediction target. Perturbing it changes the question rather than the view of it.

        Noise is in normalised units, so a std of 0.05 is 5% of a channel's training spread
        regardless of whether the channel is metres, m/s or percent.
        """
        if self.noise_std <= 0 and self.chan_drop <= 0:
            return x
        keep = np.array([c not in NO_NORM for c in self.channels])
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, x.shape).astype("float32")
            noise[:, ~keep] = 0.0
            x = x + noise
        if self.chan_drop > 0:
            # Drop whole channels for the whole sample, forcing the model to keep
            # redundant pathways rather than leaning on any single input.
            drop = (np.random.rand(len(self.channels)) < self.chan_drop) & keep
            if drop.any():
                x[:, drop] = 0.0
        return x

    def _flip(self, x: np.ndarray, target: np.ndarray, fuel: np.ndarray,
              ew: bool | None = None, ns: bool | None = None):
        """Flips, negating every channel that encodes a direction.

        This is the part of the augmentation that is easy to get silently wrong. A flip
        that moves the pixels but leaves the wind vector alone teaches mirrored physics,
        and the model trains perfectly well while learning it.

        `aspect_sin` is the east component of the downslope bearing and `aspect_cos` the
        north component (see `terrain_derivatives`), so they mirror exactly as u and v do.
        Rotation is deliberately not offered: CLAUDE.md rules it out because topography
        does not survive arbitrary rotation.

        `ew`/`ns` are explicit so the transform can be tested deterministically; left as
        None they are drawn at random, which is the training path.
        """
        ew = np.random.rand() < 0.5 if ew is None else ew
        ns = np.random.rand() < 0.5 if ns is None else ns
        ch = {c: i for i, c in enumerate(self.channels)}
        east = [ch[c] for c in ("u10_lag0", "u10_lag6", "u10_lag12", "aspect_sin") if c in ch]
        north = [ch[c] for c in ("v10_lag0", "v10_lag6", "v10_lag12", "aspect_cos") if c in ch]

        if ew:                                         # mirror east-west: flip columns
            x = x[..., ::-1].copy()
            target, fuel = target[:, ::-1].copy(), fuel[:, ::-1].copy()
            x[:, east] *= -1.0
        if ns:                                         # mirror north-south: flip rows
            x = x[..., ::-1, :].copy()
            target, fuel = target[::-1].copy(), fuel[::-1].copy()
            x[:, north] *= -1.0
        return x, target, fuel


def compute_norm_stats(cfg: dict, max_samples: int = 400, seed: int = 0) -> dict:
    """Per-channel mean/std over TRAIN fires only.

    Computing these over the whole index would leak val and test statistics into
    training. Sampled rather than exhaustive because every sample regrids weather.
    """
    ds = WildfireDataset(cfg, "train", norm_stats={}, augment=False)
    if len(ds) == 0:
        raise SystemExit("no training samples — is the index built for train-split fires?")
    rng = np.random.default_rng(seed)
    take = rng.choice(len(ds), size=min(max_samples, len(ds)), replace=False)

    n = 0
    total = np.zeros(len(ds.channels), dtype="float64")
    total_sq = np.zeros(len(ds.channels), dtype="float64")
    for i in take:
        x = ds[int(i)]["input"].numpy().astype("float64")
        total += x.sum(axis=(0, 2, 3))
        total_sq += (x**2).sum(axis=(0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]

    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean**2, 1e-12))
    stats = {}
    for i, c in enumerate(ds.channels):
        if c in NO_NORM:
            stats[c] = {"mean": 0.0, "std": 1.0, "note": "left unscaled"}
        else:
            stats[c] = {"mean": float(mean[i]), "std": float(max(std[i], 1e-6))}

    # Direction components: zero mean, one std per group. See DIRECTION_GROUPS.
    pos = {c: i for i, c in enumerate(ds.channels)}
    for group in DIRECTION_GROUPS:
        members = [c for c in group if c in pos]
        if not members:
            continue
        # Second moment about zero, pooled over the group -- not the mean of the
        # per-channel stds, which would understate the spread of a shifted channel.
        rms = float(np.sqrt(np.mean([total_sq[pos[c]] / n for c in members])))
        for c in members:
            stats[c] = {"mean": 0.0, "std": max(rms, 1e-6),
                        "note": "signed direction: zero mean, group-shared std"}
    stats["_meta"] = {"n_samples": int(len(take)), "split": "train",
                      "fires": sorted(ds.index["fire_id"].unique().tolist())}
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 dataset: splits, norm stats, checks.")
    p.add_argument("what", choices=["splits", "norm", "check"])
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=400)
    a = p.parse_args()
    cfg = load_config(a.config)

    if a.what == "splits":
        counts = write_splits(cfg)
        for k, v in counts.items():
            print(f"{k:>5}: {v:>4} fires  -> {cfg['paths']['splits']}/{k}_fires.txt")

    elif a.what == "norm":
        stats = compute_norm_stats(cfg, a.max_samples)
        out = ROOT / cfg["paths"]["norm_stats"]
        out.write_text(json.dumps(stats, indent=2))
        print(f"{'channel':<20} {'mean':>12} {'std':>12}")
        print("-" * 46)
        for c in channel_names(cfg):
            print(f"{c:<20} {stats[c]['mean']:>12.4f} {stats[c]['std']:>12.4f}")
        print(f"\nfrom {stats['_meta']['n_samples']} train samples "
              f"over fires {stats['_meta']['fires']}\nwrote {out}")

    else:
        run_checks(cfg, a.split)


def run_checks(cfg: dict, split: str) -> None:
    import time

    ds = WildfireDataset(cfg, split, augment=False)
    print(f"{split}: {len(ds)} samples over {ds.index.fire_id.nunique()} fires, "
          f"{len(ds.channels)} channels")
    if len(ds) == 0:
        return

    s = ds[0]
    print(f"  input  {tuple(s['input'].shape)}  {s['input'].dtype}")
    print(f"  fuel   {tuple(s['fuel'].shape)}  {s['fuel'].dtype}")
    print(f"  target {tuple(s['target'].shape)}  positives {s['target'].sum():.0f} px "
          f"({100*s['target'].mean():.3f}%)")
    assert torch.isfinite(s["input"]).all(), "non-finite values in input"

    # Splits must not share fires, or 50%-overlapping tiles leak across the boundary.
    sets = {k: set(read_split(cfg, k)) for k in split_years(cfg)}
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (sets[a_] & sets[b_]), f"{a_}/{b_} share fires"
    print(f"  splits disjoint: train {len(sets['train'])}, val {len(sets['val'])}, "
          f"test {len(sets['test'])} fires")

    # Flip correctness: mirroring must negate the direction channels, not merely move
    # the pixels. Driven explicitly rather than by a random draw -- with a seeded RNG it
    # is entirely possible for neither flip to fire and the test to pass vacuously.
    x = ds[0]["input"].numpy()
    tgt, fu = np.zeros((ds.patch, ds.patch), "float32"), np.zeros((ds.patch, ds.patch), "int16")
    ch = {c: i for i, c in enumerate(ds.channels)}
    ok = True
    for axis, flags, negated, kept in (
        ("east-west", dict(ew=True, ns=False), ("u10_lag0", "aspect_sin"), ("v10_lag0", "aspect_cos")),
        ("north-south", dict(ew=False, ns=True), ("v10_lag0", "aspect_cos"), ("u10_lag0", "aspect_sin")),
    ):
        fx, _, _ = ds._flip(x.copy(), tgt.copy(), fu.copy(), **flags)
        # x[:, c] is (t_steps, H, W): east-west mirrors W (last axis), north-south H.
        mirror = (lambda a: a[..., ::-1]) if flags["ew"] else (lambda a: a[:, ::-1, :])
        for c in negated:
            good = np.allclose(fx[:, ch[c]], -mirror(x[:, ch[c]]), atol=1e-4)
            ok &= good
            print(f"  flip {axis}: {c:<12} negated+mirrored -> {'OK' if good else 'MISMATCH'}"
                  f"   mean {x[:, ch[c]].mean():+.4f} -> {fx[:, ch[c]].mean():+.4f}")
        for c in kept:
            good = np.allclose(fx[:, ch[c]], mirror(x[:, ch[c]]), atol=1e-4)
            ok &= good
            print(f"  flip {axis}: {c:<12} mirrored only    -> {'OK' if good else 'MISMATCH'}")
    assert ok, "flip augmentation does not transform direction channels correctly"

    # Regression test for a real bug: flips negate the *normalised* value, so any
    # direction channel with a non-zero mean makes negation and normalisation fail to
    # commute, and the augmentation silently teaches shifted physics.
    off = [c for grp in DIRECTION_GROUPS for c in grp
           if c in ds.channels and abs(ds.mean[ds.channels.index(c)]) > 1e-9]
    assert not off, f"direction channels must have zero norm mean, got non-zero for {off}"
    print(f"  norm: {sum(len(g) for g in DIRECTION_GROUPS)} direction channels at zero mean, "
          "so flips commute with normalisation")

    t0 = time.perf_counter()
    for i in range(min(24, len(ds))):
        ds[i]
    dt = time.perf_counter() - t0
    n = min(24, len(ds))
    print(f"  throughput {n/dt:.1f} samples/s single-threaded "
          f"({1000*dt/n:.0f} ms/sample); batch 16 needs {16/(n/dt):.1f} s/batch on one worker")


if __name__ == "__main__":
    main()
