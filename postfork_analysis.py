import json, numpy as np

R = "model_trial_3/reports/dense_dump/"
idx = [json.loads(l) for l in open(R + "index.jsonl")]
gt  = np.load(R + "gt_heatmaps.npz")["gt_1s"].astype(float)   # [N,5,5]
# model alpha = direct H_1s β=2, not mse17
H   = np.load(R + "pred_beta2.npz")["H"].astype(float)        # [N,5,5]

post  = np.array([not r["pre_fork"] for r in idx])
coll  = np.array([r["outcome"] == "collision" for r in idx])
frame = np.array([r["frame"] for r in idx])

g, h = gt[post], H[post]
c, f = coll[post], frame[post]
occ = g >= 0.9
print("post-fork windows:", len(g), " occupied cells:", occ.sum())

print("\n== 1. Predictions on occupied cells (GT>=0.9): fraction per 0.1 bin, 0 -> 1 ==")
cnt, _ = np.histogram(h[occ], np.linspace(0, 1, 11))
print(np.round(cnt / cnt.sum(), 3))

print("\n== 2. Window-level failure type (windows with an occupied cell) ==")
has = occ.any((1, 2))
gmax = g.reshape(len(g), -1).argmax(1)
hmax = h.reshape(len(h), -1).argmax(1)
gr, gc = divmod(gmax, 5)
hr, hc = divmod(hmax, 5)
dist = np.maximum(abs(gr - hr), abs(gc - hc))
pmax = h.reshape(len(h), -1).max(1)
w = has
print("n windows:", w.sum())
print("missed (peak pred < 0.3):         ", round((pmax[w] < 0.3).mean(), 3))
print("peak>=0.3, argmax exact:          ", round(((pmax[w] >= 0.3) & (dist[w] == 0)).mean(), 3))
print("peak>=0.3, within 1 cell:         ", round(((pmax[w] >= 0.3) & (dist[w] == 1)).mean(), 3))
print("peak>=0.3, >=2 cells off:         ", round(((pmax[w] >= 0.3) & (dist[w] >= 2)).mean(), 3))

print("\n== 3. Side of peak cell: rows = GT side, cols = pred side [L, C, R] ==")
sd = lambda col: np.where(col <= 1, 0, np.where(col == 2, 1, 2))
conf = np.zeros((3, 3), int)
for a, b in zip(sd(gc[w]), sd(hc[w])):
    conf[a, b] += 1
print(conf)

print("\n== 4. Occupied-cell mean prediction by outcome, row, frame ==")
for name, m in [("collision", c), ("miss", ~c)]:
    mm = occ & m[:, None, None]
    print(name, round(h[mm].mean(), 3) if mm.any() else "n/a", "n", mm.sum())
for r in range(5):
    mm = occ[:, r, :]
    print("row", r, round(h[:, r, :][mm].mean(), 3) if mm.any() else "n/a", "n", mm.sum())
for fr in sorted(set(f)):
    mm = occ & (f == fr)[:, None, None]
    print("frame", fr, round(h[mm].mean(), 3) if mm.any() else "n/a", "n", mm.sum())

print("\n== 5. False hotspots: empty cells (GT<0.05) with pred > 0.3 ==")
emp = g < 0.05
print(round((h[emp] > 0.3).mean(), 4))

print("\n== 6. Plain max vs row-weighted max: post-fork collision vs miss windows ==")
# Label = clip outcome, not the heatmap. Threshold set so 5% of miss windows alert.
rw = np.array([0.5, 0.65, 0.8, 1.0, 1.0])[None, :, None]
def recall_at_fa(score, fa=0.05):
    pos, neg = score[c], score[~c]
    thr = np.quantile(neg, 1 - fa)
    return (pos > thr).mean(), thr
for name, s in [("plain max", h.reshape(len(h), -1).max(1)),
                ("row-weighted max", (h * rw).reshape(len(h), -1).max(1))]:
    for fa in (0.02, 0.05, 0.10):
        rec, thr = recall_at_fa(s, fa)
        print(f"{name:18s} FA={fa:.2f} recall={rec:.3f} thr={thr:.3f}")
