"""Training loop (Phase 6).

    python train.py                          # ConvLSTM U-Net, config defaults
    python train.py --model unet             # the capacity ablation baseline
    python train.py --smoke                  # 2 short epochs, verifies the loop end to end
    python train.py --resume checkpoints/convlstm_unet/last.pt

Follows DESIGN.md Phase 6: AMP, gradient clip at 1.0, cosine annealing with warm restarts,
per-epoch CSI and IoU on validation, and **best checkpoint by validation CSI, not loss**.
That last one matters here — loss is dominated by the 99% of pixels that are unburned, so
it keeps improving while the thing being asked for does not.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.convlstm_unet import build_model
from model.unet import UNet
from pipeline.bands import (BAND_LABELS, FAR_BAND_MIN, N_BANDS, band_pos_weights,
                            load_band_stats)
from pipeline.dataset import WildfireDataset, channel_names, read_split
from pipeline.download import ROOT, load_config
from pipeline.features import FUEL_N_CLASSES

# Probability thresholds swept on validation. With a positive rate under 1%, 0.5 is not
# obviously the right operating point, so the sweep is reported and the best is recorded
# alongside the checkpoint rather than assumed.
#
# The range must extend well above 0.5. `pos_weight` of ~103 deliberately pushes the model
# to over-predict, so its probabilities are inflated and the CSI-optimal threshold sits
# high. A first run capped at 0.7 selected 0.7 in *every* epoch — pinned at the boundary,
# with FAR still 0.81 there, which says the optimum was outside the sweep. That is not
# cosmetic: checkpoint selection and early stopping both rank epochs by this number, so a
# model whose true optimum is 0.9 would be scored at 0.7 and could lose to a worse one.
# `check_boundary` below warns if the best is ever the largest value here.
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)


@dataclass
class Counts:
    """TP/FP/FN accumulated over a split, per threshold."""

    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "Counts":
        return cls(np.zeros(n), np.zeros(n), np.zeros(n))

    def update(self, prob: torch.Tensor, target: torch.Tensor) -> None:
        t = target > 0.5
        for i, thr in enumerate(THRESHOLDS):
            p = prob > thr
            self.tp[i] += float((p & t).sum())
            self.fp[i] += float((p & ~t).sum())
            self.fn[i] += float((~p & t).sum())

    def metrics(self) -> list[dict]:
        """CSI, IoU, FAR, POD per threshold. CSI and IoU are the same quantity — both
        names are reported because the fire and vision literatures use different ones."""
        out = []
        for i, thr in enumerate(THRESHOLDS):
            tp, fp, fn = self.tp[i], self.fp[i], self.fn[i]
            csi = tp / max(tp + fp + fn, 1.0)
            out.append({
                "threshold": thr, "csi": csi, "iou": csi,
                "far": fp / max(tp + fp, 1.0),
                "pod": tp / max(tp + fn, 1.0),
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
            })
        return out


class BandCounts:
    """TP/FP/FN per distance band, at one threshold.

    Pooled CSI cannot show whether a change helped where it was aimed. Skill is
    overwhelmingly a near-perimeter effect -- CSI 0.30 at 0.2 km against 0.002 by
    2 km -- so a loss change that trades near-ring precision for far-field recall
    reads as a regression on the pooled number while doing exactly what it was
    built to do. `far_csi` pools the bands from FAR_BAND_MIN outward.
    """

    def __init__(self) -> None:
        self.tp = np.zeros(N_BANDS)
        self.fp = np.zeros(N_BANDS)
        self.fn = np.zeros(N_BANDS)

    def update(self, prob, target, band, thr: float) -> None:
        p, t = prob > thr, target > 0.5
        b = band.reshape(-1)
        for name, m in (("tp", p & t), ("fp", p & ~t), ("fn", ~p & t)):
            hit = torch.bincount(b[m.reshape(-1)], minlength=N_BANDS)
            getattr(self, name)[:] += hit.detach().cpu().numpy()

    def csi(self):
        d = self.tp + self.fp + self.fn
        return np.where(d > 0, self.tp / np.maximum(d, 1.0), np.nan)

    def far_csi(self) -> float:
        tp = self.tp[FAR_BAND_MIN:].sum()
        d = tp + self.fp[FAR_BAND_MIN:].sum() + self.fn[FAR_BAND_MIN:].sum()
        return float(tp / d) if d > 0 else 0.0


def pos_weight_from_index(cfg: dict) -> float:
    """Class-imbalance ratio over the training split, from the index alone.

    Aggregate (total unburned / total burned), not the median sample. The median sample
    is far sparser than the mean — 0.247% against 0.964% — so weighting by it implies
    ~404 instead of ~103 and would push the model to over-predict badly.
    """
    idx = pd.read_parquet(ROOT / cfg["paths"].get("sample_index", "data/processed/sample_index.parquet"))
    sub = idx[idx["fire_id"].isin(set(read_split(cfg, "train")))]
    px = int(cfg["grid"]["patch_size"]) ** 2
    pos = float(sub["n_target_px"].sum())
    total = float(len(sub) * px)
    return (total - pos) / max(pos, 1.0)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float):
    """Focal loss (DESIGN.md's fallback if class weighting is not enough).

    Computed from logits via BCE-with-logits for the same numerical reason the model
    returns logits at all.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t).pow(gamma) * bce).mean()


def make_loss(cfg: dict, device: str):
    name = str(cfg["train"].get("loss", "weighted_bce"))
    if name == "weighted_bce":
        w = cfg["train"].get("pos_weight")
        w = float(w) if w else pos_weight_from_index(cfg)
        pw = torch.tensor(w, device=device)
        # Label smoothing MULTIPLIES with pos_weight — set it with care.
        # Weighted BCE puts the optimum for a smoothed target t at
        #     p* = w*t / (w*t + 1 - t)
        # so a negative pixel with eps=0.02 (t=0.01) under w=103 is pulled to p*=0.510.
        # Measured: the reg_aug run predicted >0.5 for 98.97% of pixels, median 0.515 —
        # matching that formula — and its Brier score was 14x worse than baseline while
        # CSI looked fine, because CSI at a 0.95 threshold is blind to a shifted floor.
        # Scale eps by roughly 1/pos_weight if using it at all.
        eps = float(cfg["train"].get("label_smoothing", 0.0))
        if eps > 0 and eps * w > 0.1:
            print(f"  WARNING: label_smoothing {eps} with pos_weight {w:.0f} pulls negative "
                  f"pixels to p={w*eps/(w*eps + 1 - eps):.2f}; consider eps <= {0.1/w:.4f}")

        def loss_fn(logits, target, band=None):
            t = target * (1 - eps) + 0.5 * eps if eps > 0 else target
            return F.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)

        return loss_fn, w

    if name == "band_weighted_bce":
        # One global pos_weight is itself part of why the model builds rings. Positives
        # are ~10.1% of candidate pixels 0.1-0.2 km out but ~0.18% past 3.2 km, so the
        # balancing weight each band needs, (1-p)/p, runs from 8.9 to 552. Applying a
        # single 103 over-weights near positives ~12x and under-weights far ones ~5x,
        # which instructs the model to pile probability mass onto the perimeter.
        #
        # Rates come from the TRAIN split only (analysis/band_stats.py); taking them from
        # val or test would leak held-out statistics into the loss.
        cap = float(cfg["train"].get("band_weight_cap", 300.0))
        stats = load_band_stats(cfg)
        bw = torch.as_tensor(band_pos_weights(stats, cap), device=device)
        print(f"  per-band positive weight (train base rates, cap {cap:g}):")
        for b in range(N_BANDS):
            if int(stats["n_pixels"][b]) == 0:
                continue
            raw = (1 - stats["base_rate"][b]) / max(stats["base_rate"][b], 1e-12)
            flag = "  CAPPED" if raw > cap else ""
            print(f"    {BAND_LABELS[b]:>12}  rate {stats['base_rate'][b]:7.4%}  "
                  f"w {bw[b].item():7.1f}  (uncapped {raw:.1f}){flag}")

        def loss_fn(logits, target, band=None):
            if band is None:
                raise RuntimeError("band_weighted_bce needs the band map; set "
                                   "train.band_pos_weight: true so the Dataset emits it")
            # `weight` scales whichever BCE term is active at each pixel, so a map of
            # bw[band] on positives and 1 on negatives IS a per-pixel pos_weight.
            w_px = torch.where(target > 0.5, bw[band], torch.ones_like(target))
            return F.binary_cross_entropy_with_logits(logits, target, weight=w_px)

        # Reported as the mean weight actually applied to positives, so the startup line
        # stays comparable with the single-value runs.
        return loss_fn, float(bw[bw > 1].mean().item())
    if name == "focal":
        a = float(cfg["train"].get("focal_alpha", 0.25))
        g = float(cfg["train"].get("focal_gamma", 2.0))
        return (lambda l, t: focal_loss(l, t, a, g)), None
    raise SystemExit(f"unknown train.loss: {name!r}")


def loaders(cfg: dict, workers: int, smoke: bool):
    bs = int(cfg["train"]["batch_size"])
    tr = WildfireDataset(cfg, "train", augment=True)
    va = WildfireDataset(cfg, "val", augment=False)
    if not len(tr) or not len(va):
        raise SystemExit("empty split — run `python -m pipeline.dataset splits` first")
    common = dict(num_workers=workers, pin_memory=True,
                  persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    return (DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, **common),
            DataLoader(va, batch_size=bs, shuffle=False, **common), tr, va)


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp, limit=None):
    model.eval()
    counts = Counts.zeros(len(THRESHOLDS))
    bands: dict = {}
    total, n = 0.0, 0
    for i, b in enumerate(loader):
        if limit and i >= limit:
            break
        x, fuel, y = b["input"].to(device), b["fuel"].to(device), b["target"].to(device)
        bd = b["band"].to(device) if "band" in b else None
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x, fuel).squeeze(1)
            loss = loss_fn(logits.float(), y, bd)
        total += loss.item() * len(y)
        n += len(y)
        prob = torch.sigmoid(logits.float())
        counts.update(prob, y)
        if bd is not None:
            for thr in THRESHOLDS:
                bands.setdefault(thr, BandCounts()).update(prob, y, bd, thr)
    return total / max(n, 1), counts.metrics(), bands


def train(cfg: dict, args) -> None:
    device = str(cfg["train"].get("device", "cuda"))
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tc = cfg["train"]
    amp = bool(tc.get("amp", True))
    epochs = 2 if args.smoke else int(tc["epochs"])
    patience = int(tc.get("early_stopping_patience", 10))

    # t_horizon_h and labels.window_hours describe the same window from two files; if
    # they disagree the model is trained against a horizon it was never given data for.
    if int(cfg["model"]["t_horizon_h"]) != int(cfg["labels"]["window_hours"]):
        raise SystemExit("model.t_horizon_h must equal labels.window_hours")

    train_dl, val_dl, tr_ds, va_ds = loaders(cfg, args.workers, args.smoke)
    n_ch = len(channel_names(cfg))
    if args.model == "unet":
        model = UNet(n_ch, int(cfg["model"]["t_steps"]), FUEL_N_CLASSES,
                     int(cfg["model"].get("fuel_embed_dim", 8)),
                     tuple(cfg["model"]["hidden_dims"]),
                     supervise_centre=cfg["model"].get("supervise_centre"),
                     dropout=float(cfg["model"].get("dropout", 0.0))).to(device)
    else:
        model = build_model(cfg, FUEL_N_CLASSES, n_ch).to(device)

    loss_fn, pw = make_loss(cfg, device)
    # AdamW rather than Adam: with Adam, L2 weight decay is scaled by the per-parameter
    # adaptive step and effectively vanishes for the parameters that need it most.
    wd = float(tc.get("weight_decay", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=wd)
    # Cosine annealing with warm restarts, per DESIGN.md. T_0 in epochs.
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=int(tc.get("lr_restart_epochs", 10)), T_mult=2)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    clip = float(tc.get("grad_clip", 1.0))

    run = ROOT / "runs" / args.name
    ckpt_dir = ROOT / "checkpoints" / args.name
    run.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_csi, bad = 0, -1.0, 0
    if args.resume:
        st = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, best_csi = st["epoch"] + 1, st.get("best_csi", -1.0)
        print(f"resumed {args.resume} at epoch {start_epoch}, best CSI {best_csi:.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {params/1e6:.2f} M params, {n_ch} channels + fuel embedding")
    print(f"train {len(tr_ds):,} samples / {len(train_dl)} steps  |  val {len(va_ds):,}")
    print(f"loss {tc.get('loss')}" + (f", pos_weight {pw:.1f}" if pw else "")
          + f"  |  batch {tc['batch_size']}, lr {tc['lr']}, amp {amp}\n")

    log_path = run / "metrics.csv"
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["epoch", "train_loss", "val_loss", "csi", "iou", "far", "pod",
                         "threshold", "lr", "secs"])

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, running, seen = time.perf_counter(), 0.0, 0
        for i, b in enumerate(train_dl):
            if args.smoke and i >= 8:
                break
            x, fuel, y = b["input"].to(device, non_blocking=True), \
                b["fuel"].to(device, non_blocking=True), b["target"].to(device, non_blocking=True)
            bd = b["band"].to(device, non_blocking=True) if "band" in b else None
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x, fuel).squeeze(1)
                loss = loss_fn(logits.float(), y, bd)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Unscale before clipping or the threshold applies to scaled gradients.
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            running += loss.detach().item() * len(y)
            seen += len(y)
            if i % 25 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_dl)}  loss {loss.detach().item():.4f}",
                      flush=True)
        sched.step()

        val_loss, mets, bands = validate(model, val_dl, loss_fn, device, amp,
                                         limit=4 if args.smoke else None)
        best = max(mets, key=lambda m: m["csi"])
        at_half = next(m for m in mets if m["threshold"] == 0.5)
        if best["threshold"] in (max(THRESHOLDS), min(THRESHOLDS)):
            # The optimum is at or outside a sweep edge, so CSI is understated and epoch
            # ranking may be distorted. Widen THRESHOLDS rather than ignoring this.
            # Both edges matter: weighted_bce pushes the optimum up near 0.95, while
            # band_weighted_bce lowers the average positive weight and pulls it down
            # toward 0.2, so the bottom is the live edge for per-band runs.
            edge = "top" if best["threshold"] == max(THRESHOLDS) else "bottom"
            print(f"  WARNING: best threshold {best['threshold']} is the {edge} of the sweep")
        # The selection metric is pre-registered in the config, not picked afterwards.
        # `far_csi` exists because this loss aims at the far field, where pooled CSI is
        # nearly insensitive: 88% of all growth lies inside 1.2 km.
        sel_name = str(tc.get("select_metric", "csi"))
        if sel_name == "far_csi" and not bands:
            raise SystemExit("select_metric far_csi needs train.band_pos_weight: true")
        # Evaluate far CSI at the threshold that MAXIMISES it, not at the pooled-best
        # one. The two differ a lot -- the pooled optimum is set by the near ring and
        # runs high, and reading far CSI there understates it badly: the baseline scores
        # 0.0514 at its pooled-best 0.95 against 0.0863 at 0.85. Selecting on a metric
        # while measuring it at another objective's operating point is the same error
        # as the 0.7-capped threshold sweep that once inverted the epoch ranking.
        far_thr = max(THRESHOLDS, key=lambda t: bands[t].far_csi()) if bands else None
        far_csi = bands[far_thr].far_csi() if bands else float("nan")
        sel = far_csi if sel_name == "far_csi" else best["csi"]

        secs = time.perf_counter() - t0
        if bands:
            bc = bands[best["threshold"]].csi()
            print("  CSI by distance: " + "  ".join(
                f"{BAND_LABELS[b].split()[0]} {bc[b]:.3f}"
                for b in range(N_BANDS) if not np.isnan(bc[b])), flush=True)
            print(f"  far CSI (>= 1.2 km) {far_csi:.4f} @thr {far_thr}   "
                  f"pooled {best['csi']:.4f} @thr {best['threshold']}", flush=True)
        print(f"epoch {epoch}: train {running/max(seen,1):.4f}  val {val_loss:.4f}  "
              f"CSI {best['csi']:.4f} @thr {best['threshold']}  "
              f"(CSI@0.5 {at_half['csi']:.4f})  FAR {best['far']:.3f}  POD {best['pod']:.3f}  "
              f"{secs/60:.1f} min", flush=True)
        writer.writerow([epoch, running/max(seen,1), val_loss, best["csi"], best["iou"],
                         best["far"], best["pod"], best["threshold"],
                         opt.param_groups[0]["lr"], round(secs, 1)])
        log.flush()
        (run / "last_metrics.json").write_text(json.dumps(mets, indent=2))

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch, "best_csi": best_csi,
                 "threshold": best["threshold"], "cfg_channels": channel_names(cfg),
                 "model_kind": args.model}
        torch.save(state, ckpt_dir / "last.pt")
        if sel > best_csi:
            best_csi, bad = sel, 0
            state["best_csi"] = best_csi
            state["select_metric"] = sel_name
            state["far_csi"] = far_csi
            state["far_threshold"] = far_thr
            state["pooled_csi"] = best["csi"]
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  new best {sel_name} {best_csi:.4f} -> {ckpt_dir/'best.pt'}")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop: {patience} epochs without {sel_name} improvement")
                break

    log.close()
    print(f"\nbest validation {str(tc.get('select_metric', 'csi'))} {best_csi:.4f}"
          f"  |  {ckpt_dir/'best.pt'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the wildfire spread model.")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--model", choices=["convlstm_unet", "unet"], default="convlstm_unet")
    p.add_argument("--name", help="run name (defaults to --model)")
    p.add_argument("--workers", type=int,
                   help="DataLoader workers (default: train.num_workers in the config)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dims", help="override model.hidden_dims, e.g. 112,224,448")
    p.add_argument("--resume")
    p.add_argument("--smoke", action="store_true", help="2 tiny epochs, checks the loop")
    a = p.parse_args()
    a.name = a.name or a.model
    cfg = load_config(a.config)
    if a.hidden_dims:
        cfg["model"]["hidden_dims"] = [int(v) for v in a.hidden_dims.split(",")]
    if a.workers is None:
        a.workers = int(cfg["train"].get("num_workers", 6))
    train(cfg, a)


if __name__ == "__main__":
    main()
