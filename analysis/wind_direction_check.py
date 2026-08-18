"""Is growth direction related to wind direction IN THE DATA?

Precondition for every fix aimed at making the model use HRRR. If the labels do
not respond to wind, no architecture or loss change can create the signal.

Convention: HRRR u = eastward, v = northward. The wind VECTOR (u,v) points where
air is going, and fire runs downwind, so growth should align with (u,v).
Rasters are north-up, so row increases southward: (east, north) = (+dcol, -drow).
"""
import numpy as np, yaml, sys
from pipeline.dataset import WildfireDataset

cfg = yaml.safe_load(open("configs/baseline.yaml", encoding="utf-8"))
ds = WildfireDataset(cfg, "train", norm_stats={}, augment=False)   # raw units
ch = {c: i for i, c in enumerate(ds.channels)}
iu, iv, ib = ch["u10_lag0"], ch["v10_lag0"], ch["burn_mask"]
print(f"{len(ds)} train samples; using channels u={iu} v={iv} burn={ib}", flush=True)

rng = np.random.default_rng(0)
take = rng.choice(len(ds), size=min(600, len(ds)), replace=False)

rows = []
for n, i in enumerate(take):
    s = ds[int(i)]
    x, tgt = np.asarray(s["input"]), np.asarray(s["target"])
    cur = x[-1, ib] > 0.5                      # most recent step = current extent
    grow = tgt > 0.5
    if cur.sum() < 20 or grow.sum() < 20:
        continue
    rr, cc = np.nonzero(cur);   cur_row, cur_col = rr.mean(), cc.mean()
    gr, gc = np.nonzero(grow);  gro_row, gro_col = gr.mean(), gc.mean()
    de, dn = (gro_col - cur_col), -(gro_row - cur_row)   # growth vector, east/north
    mag = float(np.hypot(de, dn))
    if mag < 1e-6:
        continue
    u = float(x[-1, iu].mean()); v = float(x[-1, iv].mean())
    spd = float(np.hypot(u, v))
    if spd < 1e-6:
        continue
    # angle between growth vector and wind vector, degrees, 0 = perfectly downwind
    cos = (de*u + dn*v) / (mag*spd)
    rows.append((np.degrees(np.arccos(np.clip(cos, -1, 1))), mag, spd, grow.sum(), cos))
    if n % 100 == 0: print(f"  {n}/{len(take)}", flush=True)

a = np.array(rows)
print(f"\nusable samples: {len(a)}")
ang, mag, spd, gsz, cos = a[:,0], a[:,1], a[:,2], a[:,3], a[:,4]
print(f"\nangle between growth and wind (0 deg = downwind):")
for lo, hi in [(0,45),(45,90),(90,135),(135,180)]:
    f = ((ang>=lo)&(ang<hi)).mean()
    print(f"  {lo:3d}-{hi:3d} deg : {f:6.1%}   ({'downwind' if lo==0 else 'upwind' if lo==135 else 'cross'})")
print(f"\n  mean angle      {ang.mean():.1f} deg   (90 = no relationship)")
print(f"  mean cos        {cos.mean():+.4f}  (0 = no relationship, +1 = perfect)")
print(f"  median wind spd {np.median(spd):.2f} m/s")

# Does the relationship strengthen when wind is strong / growth is large?
for name, key, qs in [("wind speed", spd, None), ("growth size", gsz, None)]:
    print(f"\n  by {name} quartile:  mean cos")
    q = np.quantile(key, [0,.25,.5,.75,1.0])
    for j in range(4):
        m = (key>=q[j]) & (key<=q[j+1] if j==3 else key<q[j+1])
        if m.sum(): print(f"    Q{j+1} (n={m.sum():3d}): {cos[m].mean():+.4f}")
