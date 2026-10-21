import json, numpy as np
R = "model_trial_3/reports/dense_dump/"
idx = [json.loads(l) for l in open(R + "index.jsonl")]
gt  = np.load(R + "gt_heatmaps.npz")["gt_1s"].astype(float)
H   = np.load(R + "pred_beta2.npz")["H"].astype(float)      # model alpha

post  = np.array([not r["pre_fork"] for r in idx])
coll  = np.array([r["outcome"] == "collision" for r in idx])
amb   = np.array([r["amb_1s"] for r in idx], bool)
frame = np.array([r["frame"] for r in idx])
clip  = np.array([r["clip_id"] for r in idx])

g, h, c, f, cl = gt[post], H[post], coll[post], frame[post], clip[post]
N = len(g)
gm, hm = g.reshape(N, -1), h.reshape(N, -1)
gmax, hmax = gm.max(1), hm.max(1)

def rec(score, mask, fa):
    pos, neg = score[mask & c], score[mask & ~c]
    if len(pos) == 0 or len(neg) == 0: return None
    thr = np.quantile(neg, 1 - fa)
    return round((pos > thr).mean(), 3), round((neg > thr).mean(), 3), round(float(thr), 3)

print("== A. Ceiling check: recall (actual FA) for collision vs miss, post-fork ==")
allw = np.ones(N, bool)
for name, s in [("model max", hmax), ("GT-heatmap max (oracle)", gmax)]:
    for fa in (0.02, 0.05, 0.10):
        print(f"{name:26s} FA={fa:.2f} -> (recall, actual FA, thr)", rec(s, allw, fa))

print("\n== B. Only windows where GT has an occupied cell (>=0.9) ==")
sub = gmax >= 0.9
print("collision windows:", (sub & c).sum(), " miss windows:", (sub & ~c).sum())
for fa in (0.05, 0.10):
    print("model max FA", fa, rec(hmax, sub, fa))

print("\n== C. Localization using the whole occupied set (not one argmax) ==")
occ = gm >= 0.9
w = occ.any(1)
print("windows:", w.sum(), " mean occupied cells/window:", round(occ[w].sum(1).mean(), 2))
peak = hm.argmax(1)
inset = occ[np.arange(N), peak]
print("peak cell lies inside occupied set:", round(inset[w].mean(), 3))
side = lambda col: np.where(col <= 1, 0, np.where(col == 2, 1, 2))
cols = np.arange(25) % 5
ps = side(peak % 5)
gs_has = np.stack([(occ & (side(cols) == k)[None, :]).any(1) for k in range(3)], 1)
print("pred-peak side has GT occupancy:", round(gs_has[np.arange(N), ps][w].mean(), 3))
opp = ((ps == 0) & gs_has[:, 2] & ~gs_has[:, 0] & ~gs_has[:, 1]) | \
      ((ps == 2) & gs_has[:, 0] & ~gs_has[:, 1] & ~gs_has[:, 2])
print("opposite-side errors (L<->R only):", opp[w].sum(), "of", w.sum())
hot = hm >= 0.5
tp = (hot & occ).sum(1)
print("cell precision @0.5:", round((tp[w] / np.maximum(1, hot[w].sum(1))).mean(), 3),
      " cell recall @0.5:", round((tp[w] / occ[w].sum(1)).mean(), 3))

print("\n== D. Error by frame, ALL rows (* = on stride-15 grid from 90) ==")
fr_all = frame; err_all = ((H - gt) ** 2).mean((1, 2))
for fr in sorted(set(fr_all)):
    m = (fr_all == fr) & ~amb
    mark = "*" if (fr - 90) % 15 == 0 else " "
    print(fr, mark, round(err_all[m].mean(), 4))
on = ((fr_all - 90) % 15 == 0) & ~amb
off = ((fr_all - 90) % 15 != 0) & ~amb
print("mean MSE on-grid:", round(err_all[on].mean(), 4), " off-grid:", round(err_all[off].mean(), 4))

print("\n== E. Flicker: post-fork, per clip, threshold 0.5 on max ==")
sw, dd, gsw = [], [], []
for k in set(cl):
    m = np.where(cl == k)[0]; m = m[np.argsort(f[m])]
    a = hmax[m] >= 0.5; b = gmax[m] >= 0.5
    sw.append((a[1:] != a[:-1]).sum()); gsw.append((b[1:] != b[:-1]).sum())
    dd.append(np.abs(np.diff(hmax[m])).mean())
print("model switches/clip:", round(np.mean(sw), 2), " GT switches/clip:", round(np.mean(gsw), 2),
      " mean |diff| of max between anchors:", round(np.mean(dd), 3))
