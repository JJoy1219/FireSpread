"""Where does the model's skill actually live, as a function of distance from the
current perimeter?

Sets the weights for any distance-weighted loss, and defines the metric that such
a loss would have to be judged by. Bins every test pixel by its euclidean distance
(in 100 m cells) to the nearest currently-burned cell, then reports per band:
base rate of real growth, the model's CSI, and what share of all growth lives there.
"""
import numpy as np, torch, yaml
from scipy.ndimage import distance_transform_edt
from pipeline.dataset import WildfireDataset
from pipeline.viz_predict import load_model

cfg = yaml.safe_load(open("configs/baseline.yaml", encoding="utf-8"))
dev = "cpu"
cfg["model"]["hidden_dims"] = [112, 224, 448]   # load_model reads dims from cfg
model, thr, _ = load_model(cfg, "unet", "checkpoints/unet_wide/best.pt", dev)
model.eval()
ds = WildfireDataset(cfg, "test", augment=False)
ib = {c: i for i, c in enumerate(ds.channels)}["burn_mask"]
print(f"test {len(ds)} samples, threshold {thr}", flush=True)

EDGES = [0, 1, 2, 3, 5, 8, 12, 20, 32, 1e9]         # cells; 1 cell = 100 m
NB = len(EDGES) - 1
tp = np.zeros(NB); fp = np.zeros(NB); fn = np.zeros(NB); npx = np.zeros(NB); pos = np.zeros(NB)

rng = np.random.default_rng(0)
take = rng.choice(len(ds), size=min(200, len(ds)), replace=False)
for n, i in enumerate(take):
    s = ds[int(i)]
    x = s["input"][None].to(dev)
    with torch.no_grad():
        p = torch.sigmoid(model(x, s["fuel"][None].to(dev))).cpu().numpy()[0, 0]
    tgt = s["target"].numpy() > 0.5
    cur = s["input"][-1, ib].numpy() > 0.5
    if cur.sum() == 0:
        continue
    d = distance_transform_edt(~cur)                  # distance to nearest burned cell
    pred = p >= thr
    for b in range(NB):
        m = (d >= EDGES[b]) & (d < EDGES[b+1])
        if not m.any(): continue
        tp[b] += (pred & tgt & m).sum(); fp[b] += (pred & ~tgt & m).sum()
        fn[b] += (~pred & tgt & m).sum(); npx[b] += m.sum(); pos[b] += (tgt & m).sum()
    if n % 50 == 0: print(f"  {n}/{len(take)}", flush=True)

print(f"\n{'dist (cells)':>14} {'km':>6} {'base rate':>10} {'CSI':>7} {'POD':>6} {'FAR':>6} {'% of all growth':>16}")
tot = pos.sum()
for b in range(NB):
    if npx[b] == 0: continue
    lo, hi = EDGES[b], EDGES[b+1]
    lbl = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
    km = f"{lo*0.1:.1f}+" if hi > 1e8 else f"{hi*0.1:.1f}"
    csi = tp[b]/(tp[b]+fp[b]+fn[b]) if (tp[b]+fp[b]+fn[b]) else 0
    pod = tp[b]/(tp[b]+fn[b]) if (tp[b]+fn[b]) else 0
    far = fp[b]/(tp[b]+fp[b]) if (tp[b]+fp[b]) else float('nan')
    print(f"{lbl:>14} {km:>6} {pos[b]/npx[b]:9.3%} {csi:7.4f} {pod:6.3f} {far:6.3f} {pos[b]/tot:15.1%}")
print(f"\noverall CSI {tp.sum()/(tp.sum()+fp.sum()+fn.sum()):.4f}   total growth px {tot:,.0f}")
