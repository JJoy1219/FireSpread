"""Prediction visualisation: what the model actually got right and wrong.

    python -m pipeline.viz_predict cases  --checkpoint checkpoints/unet_wide/best.pt --model unet
    python -m pipeline.viz_predict fire   --fire-id 2022_3298
    python -m pipeline.viz_predict calib

Three views, each answering a question the scalar metrics cannot:

* `cases`  — best and worst samples by CSI, as confusion maps. Shows *how* it fails
             (wrong direction? over-spread? missed a run?) rather than only how much.
* `fire`   — one fire across consecutive windows, so errors can be read as a sequence.
* `calib`  — reliability diagram. CSI is threshold-dependent and says nothing about
             whether the probabilities mean anything; Brier alone does not show *where*
             calibration breaks.

The confusion map is the workhorse. Predicted-vs-truth as two panels invites eyeballing
overlap, which is exactly what humans do badly; TP/FP/FN as one colour-coded field makes
the error structure immediate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from model.convlstm_unet import build_model
from model.unet import UNet
from pipeline.dataset import WildfireDataset, channel_names, read_split
from pipeline.download import ROOT, load_config
from pipeline.features import FUEL_N_CLASSES

# Colour-blind-safe and deliberately asymmetric: false alarms and misses are different
# operational failures and must not be confusable at a glance.
CONF_COLOURS = ["#f0f0f0", "#2c7fb8", "#d95f02", "#1b9e77"]   # none, FN(miss), FP(alarm), TP
CONF_LABELS = ["correct negative", "missed burn (FN)", "false alarm (FP)", "hit (TP)"]


def load_model(cfg: dict, kind: str, ckpt: str, device: str):
    n_ch = len(channel_names(cfg))
    if kind == "unet":
        m = UNet(n_ch, int(cfg["model"]["t_steps"]), FUEL_N_CLASSES,
                 int(cfg["model"].get("fuel_embed_dim", 8)),
                 tuple(cfg["model"]["hidden_dims"])).to(device)
    else:
        m = build_model(cfg, FUEL_N_CLASSES, n_ch).to(device)
    st = torch.load(ROOT / ckpt, map_location=device, weights_only=False)
    m.load_state_dict(st["model"])
    m.eval()
    return m, float(st.get("threshold", 0.5)), st


def confusion(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """0 none, 1 FN, 2 FP, 3 TP."""
    return (truth & ~pred) * 1 + (pred & ~truth) * 2 + (pred & truth) * 3


def csi_of(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = float((pred & truth).sum())
    return tp / max(tp + float((pred & ~truth).sum()) + float((~pred & truth).sum()), 1.0)


@torch.no_grad()
def run_batch(model, ds, rows, device, amp=True):
    """Predict a list of dataset indices. Returns (prob, current_burn, target)."""
    xs, fs, ys, cur = [], [], [], []
    for i in rows:
        s = ds[int(i)]
        xs.append(s["input"]); fs.append(s["fuel"]); ys.append(s["target"])
        # Burn mask of the most recent step, before normalisation touched anything.
        cur.append(s["input"][-1, ds.channels.index("burn_mask")].numpy())
    x = torch.stack(xs).to(device); f = torch.stack(fs).to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
        prob = torch.sigmoid(model(x, f).squeeze(1).float())
    return prob.cpu().numpy(), np.stack(cur), torch.stack(ys).numpy()


def draw_case(ax_row, prob, cur, truth, thr, title):
    """One sample: current burn, probability, confusion map."""
    pred = prob > thr
    t = truth > 0.5
    a0, a1, a2 = ax_row

    a0.imshow(cur, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    a0.set_title(f"{title}\nburned so far", fontsize=8)

    im = a1.imshow(prob, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    # Truth outline over the probability field: shows whether high probability lands in
    # the right place, which two side-by-side masks do not make obvious.
    a1.contour(t.astype(float), levels=[0.5], colors="#00d4ff", linewidths=0.8)
    a1.set_title(f"P(new burn), truth outlined\nmax {prob.max():.2f}", fontsize=8)

    a2.imshow(confusion(pred, t), cmap=ListedColormap(CONF_COLOURS), vmin=0, vmax=3,
              interpolation="nearest")
    tp, fp, fn = (pred & t).sum(), (pred & ~t).sum(), (~pred & t).sum()
    a2.set_title(f"CSI {csi_of(pred, t):.3f}  @thr {thr}\nTP {tp}  FP {fp}  FN {fn}", fontsize=8)
    for a in ax_row:
        a.set_xticks([]); a.set_yticks([])
    return im


def _finish(fig, im, title: str) -> None:
    """Lay out, then add the colourbar on a dedicated axis.

    `fig.colorbar(ax=...)` steals space from the panels it is given, which tight_layout
    then cannot account for — the bar ends up on top of the images. Reserving an explicit
    axis after layout keeps the panels square and the bar clear of them.
    """
    fig.tight_layout(rect=[0, 0.045, 0.90, 0.97])
    cax = fig.add_axes([0.92, 0.10, 0.018, 0.80])
    fig.colorbar(im, cax=cax, label="P(new burn)")
    fig.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                               markersize=9, label=l)
                        for c, l in zip(CONF_COLOURS, CONF_LABELS)],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)


def plot_cases(cfg, model, thr, ds, device, out: Path, n: int = 4) -> Path:
    """Best and worst samples by CSI — the failure modes, not just the average."""
    idx = np.random.default_rng(0).choice(len(ds), size=min(240, len(ds)), replace=False)
    scores = []
    for start in range(0, len(idx), 12):
        chunk = idx[start:start + 12]
        prob, cur, y = run_batch(model, ds, chunk, device)
        for k, i in enumerate(chunk):
            scores.append((csi_of(prob[k] > thr, y[k] > 0.5), int(i)))
    scores.sort()
    picks = [i for _, i in scores[-n:][::-1]] + [i for _, i in scores[:n]]
    labels = [f"BEST {j+1}" for j in range(n)] + [f"WORST {j+1}" for j in range(n)]

    prob, cur, y = run_batch(model, ds, picks, device)
    fig, axes = plt.subplots(len(picks), 3, figsize=(8.2, 2.7 * len(picks)))
    for r, (p, c, t, lab) in enumerate(zip(prob, cur, y, labels)):
        meta = ds.index.iloc[picks[r]]
        im = draw_case(axes[r], p, c, t, thr, f"{lab}  {meta.fire_id} t={meta.t_index}")
    _finish(fig, im, f"Best and worst {ds.split} samples by CSI")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_fire(cfg, model, thr, ds, device, fire_id: str, out: Path, n: int = 5) -> Path:
    """One fire across consecutive windows — errors read as a sequence, not snapshots."""
    sub = ds.index[ds.index.fire_id == fire_id]
    if sub.empty:
        raise SystemExit(f"{fire_id} has no samples in the {ds.split} split")
    # One tile position, walked forward in time, so panels are comparable.
    tile = sub.groupby(["row0", "col0"]).size().idxmax()
    seq = sub[(sub.row0 == tile[0]) & (sub.col0 == tile[1])].sort_values("t_index")
    picks = seq.index[:n].tolist()

    prob, cur, y = run_batch(model, ds, picks, device)
    fig, axes = plt.subplots(len(picks), 3, figsize=(8.2, 2.7 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(picks):
        meta = ds.index.iloc[i]
        im = draw_case(axes[r], prob[r], cur[r], y[r], thr,
                       f"{fire_id}  window {meta.t_index}")
    _finish(fig, im, f"{fire_id}: consecutive 24 h windows, one tile")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_calibration(cfg, model, ds, device, out: Path, n_samples: int = 240) -> Path:
    """Reliability diagram: do the probabilities mean what they say?

    CSI is computed at one threshold and is blind to calibration; Brier is a single
    number and cannot show *where* it breaks. With `pos_weight` ~103 pushing the model to
    over-predict, systematic overconfidence is the expected failure and this is where it
    becomes visible.
    """
    idx = np.random.default_rng(0).choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    edges = np.array([0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0])
    tot = np.zeros(len(edges) - 1)
    hit = np.zeros(len(edges) - 1)
    psum = np.zeros(len(edges) - 1)
    for start in range(0, len(idx), 12):
        prob, _, y = run_batch(model, ds, idx[start:start + 12], device)
        b = np.clip(np.digitize(prob.ravel(), edges) - 1, 0, len(edges) - 2)
        np.add.at(tot, b, 1)
        np.add.at(hit, b, (y.ravel() > 0.5).astype(float))
        np.add.at(psum, b, prob.ravel())
    ok = tot > 0
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="perfect")
    ax.plot(psum[ok] / tot[ok], hit[ok] / tot[ok], "o-", color="#d95f02", label="model")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed burn frequency")
    ax.set_title("Reliability"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax2.bar(range(ok.sum()), tot[ok], color="#2c7fb8")
    ax2.set_yscale("log"); ax2.set_xlabel("probability bin"); ax2.set_ylabel("pixels (log)")
    ax2.set_title("Bin occupancy — most pixels sit near zero")
    ax2.set_xticks(range(ok.sum()))
    ax2.set_xticklabels([f"{e:g}" for e in edges[:-1][ok]], rotation=60, fontsize=6)
    fig.suptitle(f"Calibration on {ds.split} ({len(idx)} samples)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


CHANNEL_GROUPS = {
    "wind (u/v, all lags)": ["u10_lag0", "u10_lag6", "u10_lag12",
                             "v10_lag0", "v10_lag6", "v10_lag12"],
    "moisture (RH, Fosberg)": ["rh2m", "fosberg_10h"],
    "temperature": ["t2m"],
    "terrain (slope/aspect/TPI/elev)": ["elevation_centred", "slope", "aspect_sin",
                                        "aspect_cos", "tpi"],
    "canopy (CC, CH)": ["cc", "ch"],
    "burn mask": ["burn_mask"],
}


@torch.no_grad()
def plot_sensitivity(cfg, model, thr, ds, device, out: Path, n_samples: int = 96,
                     mode: str = "zero") -> Path:
    """Occlusion test: how much does each input group actually change the prediction?

    Motivated by the case figure, where predictions look like near-symmetric halos around
    the current perimeter rather than wind-driven plumes. If zeroing the wind barely moves
    the output, the model is a learned dilation and the HRRR pipeline is not earning its
    place in the input.

    Each group is set to its normalised mean (0 after normalisation, so zeroing IS the
    mean-imputation) and the prediction is recomputed. Reported as mean |delta P| over
    pixels near the fire, and as the change in CSI. Pixels far from any fire are excluded:
    they are ~99% of the tile, always near zero, and would dilute every effect to nothing.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    ch = {c: i for i, c in enumerate(ds.channels)}
    rows = []
    base_csi_all, deltas = [], {k: [] for k in CHANNEL_GROUPS}
    csis = {k: [] for k in CHANNEL_GROUPS}

    for start in range(0, len(idx), 12):
        chunk = idx[start:start + 12]
        xs, fs, ys = [], [], []
        for i in chunk:
            sm = ds[int(i)]
            xs.append(sm["input"]); fs.append(sm["fuel"]); ys.append(sm["target"])
        x = torch.stack(xs).to(device); f = torch.stack(fs).to(device)
        y = torch.stack(ys).numpy() > 0.5
        base = torch.sigmoid(model(x, f).squeeze(1).float()).cpu().numpy()
        # Region of interest: within ~25 cells of somewhere the model or truth is active.
        roi = (base > 0.01) | y
        for k in range(len(chunk)):
            base_csi_all.append(csi_of(base[k] > thr, y[k]))

        for name, cols in CHANNEL_GROUPS.items():
            xm = x.clone()
            # `zero` sets the channel to its normalised mean (mean-imputation).
            # `permute` shuffles it across the batch instead: this destroys the link to
            # the target while preserving a realistic marginal distribution, so the model
            # cannot benefit from recognising an artificial constant. Permutation is the
            # stronger test and the one to trust when the two disagree.
            perm = torch.randperm(xm.shape[0], device=xm.device)
            for c in cols:
                if c in ch:
                    if mode == "permute":
                        xm[:, :, ch[c]] = xm[perm][:, :, ch[c]]
                    else:
                        xm[:, :, ch[c]] = 0.0
            alt = torch.sigmoid(model(xm, f).squeeze(1).float()).cpu().numpy()
            for k in range(len(chunk)):
                m = roi[k]
                deltas[name].append(float(np.abs(alt[k] - base[k])[m].mean()) if m.any() else 0.0)
                csis[name].append(csi_of(alt[k] > thr, y[k]))

    base_csi = float(np.mean(base_csi_all))
    for name in CHANNEL_GROUPS:
        rows.append((name, float(np.mean(deltas[name])),
                     float(np.mean(csis[name])), base_csi - float(np.mean(csis[name]))))
    rows.sort(key=lambda r: -r[1])

    print(f"\nbaseline CSI {base_csi:.4f} over {len(idx)} {ds.split} samples\n")
    print(f"{'group zeroed':<34}{'mean |dP| in ROI':>18}{'CSI':>9}{'CSI drop':>10}")
    print("-" * 71)
    for name, d, c, drop in rows:
        print(f"{name:<34}{d:>18.4f}{c:>9.4f}{drop:>+10.4f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [r[0] for r in rows]
    ax.barh(names, [r[1] for r in rows], color="#2c7fb8")
    ax.invert_yaxis(); ax.set_xlabel("mean |change in P| near the fire")
    ax.set_title(f"Prediction sensitivity ({mode} each input group)")
    ax.grid(alpha=.3, axis="x")
    ax2.barh(names, [r[3] for r in rows], color="#d95f02")
    ax2.invert_yaxis(); ax2.set_xlabel("CSI lost when group is removed")
    ax2.axvline(0, color="0.4", lw=1)
    ax2.set_title(f"Skill cost (baseline CSI {base_csi:.3f})")
    ax2.grid(alpha=.3, axis="x")
    fig.suptitle(f"Input occlusion on {ds.split} ({len(idx)} samples)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise model predictions against truth.")
    p.add_argument("what", choices=["cases", "fire", "calib", "sensitivity"])
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--model", choices=["unet", "convlstm_unet"], default="unet")
    p.add_argument("--checkpoint", default="checkpoints/unet_wide/best.pt")
    p.add_argument("--hidden-dims", default="112,224,448")
    p.add_argument("--split", default="test")
    p.add_argument("--fire-id", default="2022_3298")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--mode", choices=["zero", "permute"], default="zero",
                   help="sensitivity: mean-impute or shuffle across the batch")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if free)")
    p.add_argument("--out")
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.hidden_dims:
        cfg["model"]["hidden_dims"] = [int(v) for v in a.hidden_dims.split(",")]
    # Default to CPU when the GPU is busy training — these figures are small and the
    # point is to be able to look at predictions without disturbing a run.
    device = a.device or ("cuda" if torch.cuda.is_available()
                          and torch.cuda.mem_get_info()[0] / 2**30 > 4 else "cpu")
    model, thr, st = load_model(cfg, a.model, a.checkpoint, device)
    ds = WildfireDataset(cfg, a.split, augment=False)
    print(f"{a.checkpoint}  epoch {st.get('epoch')}  threshold {thr}  device {device}")
    print(f"{a.split}: {len(ds)} samples over {ds.index.fire_id.nunique()} fires")

    figs = ROOT / "figures"
    if a.what == "cases":
        out = plot_cases(cfg, model, thr, ds, device,
                         Path(a.out) if a.out else figs / f"pred_cases_{a.split}.png", a.n)
    elif a.what == "fire":
        out = plot_fire(cfg, model, thr, ds, device, a.fire_id,
                        Path(a.out) if a.out else figs / f"pred_fire_{a.fire_id}.png", a.n)
    elif a.what == "sensitivity":
        out = plot_sensitivity(cfg, model, thr, ds, device,
                               Path(a.out) if a.out else figs / f"pred_sens_{a.split}_{a.mode}.png",
                               mode=a.mode)
    else:
        out = plot_calibration(cfg, model, ds, device,
                               Path(a.out) if a.out else figs / f"pred_calib_{a.split}.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
