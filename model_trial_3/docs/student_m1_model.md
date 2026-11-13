# Student M1 (STOP-floor / overnight `m1_stop_floor`)

**Current best RGB student** under the official selection rule
(SAFE ≥ 0.85, then max STOP recall, then STOP precision).

| | |
|---|---|
| Run dir | `model_trial_3_temp/overnight6/m1_stop_floor/` |
| Official ckpt | `checkpoint_epoch_10.pt` (`checkpoint_best.pt`) |
| Seed | 1001 |
| Parent decoder hash at init | `77f78c8714667617` |
| Params | **980 171** total (encoder 881 290, predictor 34 560, 1×1 proj 16 512, decoder 47 809) |
| CPU latency | ~**37 ms** / clip (float32) |
| Compute (earlier audit) | ~1.79 GMAC / ~3.59 GFLOP |

M1 does **not** run V-JEPA. It maps 16 RGB frames → parent-shaped
`F` `[B,128,16,16]` → the **copied Pair-CF decoder** → 5×5 heatmap.

---

## 1. Role relative to the parent

The parent is frozen Giant + adapter + decoder, trained on GT occupancy and
Pair-CF shrink (see `parent_pair_cf_model.md`).

M1 is a **tiny Conv3D student** whose job is:

1. Match cached parent **`F_dec`** (main loss)
2. Put enough **amplitude on GT-STOP** maps (STOP floor at 0.70)
3. Stay close to GT occupancy and to parent heatmaps **when parent agrees
   with GT**
4. Keep the decoder near Pair-CF init while slowly unfreezing it

It is trained **from random student weights** (identity 1×1 proj). Decoder
starts as Pair-CF, not from another student.

---

## 2. Data (same contract as later students)

- RGB cache: `model_trial_3_temp/shared_rgb_cache_256_v1/` (letterbox 256)
- Parent cache: `parent_target_cache_v2` (`F_dec`, `H_teacher`)
- Splits: **22 680 / 5 120 / 5 120** train / selection / test
- **Test is never loaded** for train or checkpoint choice
- Offsets: same 16-frame parent offsets
- GT: **group-mean** 5×5 `gt_1s`
- Class-balanced sampling is **not** used in the overnight trainer
  (`shuffle=True`, batch 8)

Warning thresholds are fixed: SAFE \(<0.25\), STOP \(\ge 0.60\).

---

## 3. Architecture

Class `StudentOvernight` with `variant=m1_stop_floor` (no aux head, no
recent-frame branch — those are M5/M6).

```
clip [B,16,3,256,256]
  ImageNet normalize, permute to [B,3,16,256,256]
  StudentEncoder (2+1)D
      stem: Conv3d 3→64, k=(2,16,16), s=(2,16,16)   → tubelet grid
      2× inverted residual (width 64, expand 4)
      1×1 to 128 ch
      6× inverted residual (expand 3, dilations 1,2,1,2,1,2)
      extra residual (expand 2)
      TemporalSoftmaxPool over T' plus a learned last-frame gate
  → F_ctx [B,128,16,16]
  ChannelProj 1×1  (init = identity)
  → F      [B,128,16,16]     # fed to decoder; distilled to parent F_dec
  JEPAPredictor(128, grid=16) on F_ctx   # masked feature reconstruction
  Pair-CF HeatmapDecoder.forward_with_logits(F)
  → Hhat_1s, logits [B,5,5]
```

**Inverted residual:** pointwise expand → spatial DW 3×3 → temporal DW 3×1
(dilation) → project + GroupNorm + residual GELU.

**ChannelProj:** 1×1 conv, weights = \(I_{128}\) at init so early `F ≈ F_ctx`.

**Predictor:** small masked CNN used only for \(L_{mgd}\), not for the
heatmap path.

**Not in M1:** aux 3-way classifier (M6), dual recent-4-frame branch (M5).

---

## 4. Loss (exact overnight M1)

Let \(F_s\) be student `F`, \(F_t\) cached parent `F_dec`, both
\([B,128,16,16]\). Let \(\hat H, Y, H_t\) be student heatmap, GT, parent
heatmap on \(5\times 5\). Decoder logits exist but **M1 never uses them**.
Peak class: SAFE if \(\max < 0.25\), THREAT if \(< 0.60\), else STOP.

M1 does **not** use parent BCE, appear×8, blob, STOP/SAFE hinges, or
Pair-CF PCA. Those live only on the parent.

---

### 4.1 Spatial weights on the 16×16 grid

`blob_weights(F_t, h_now=None, gt_1s, energy_alpha=0, appear_boost=0, stop_boost=0)`:

\[
E_{bhw} = \frac{1}{128}\sum_c (F_t)_{bchw}^2,
\quad
\tilde E = E / \mathrm{mean}_{hw}(E).
\]

Full formula is \(w = 1 + \alpha \tilde E + 2\cdot\mathbf{1}[Y_{\uparrow}\ge 0.60] + 4\cdot\mathbf{1}[\mathrm{appear}]\),
then mean-normalized. Overnight M1 sets **\(\alpha=\)appear=stop boosts = 0**,
so after normalization \(w \equiv 1\). The GT map is passed in but does not
change \(w\). Every spatial cell counts equally.

---

### 4.2 Channel whitening

Cached per-channel `mu`, `sigma` from parent `F_dec` train stats
(`[128,1,1]`):

\[
\bar F_s = \frac{F_s - \mu}{\sigma + 10^{-6}},
\quad
\bar F_t = \frac{F_t - \mu}{\sigma + 10^{-6}}.
\]

Stops a few high-variance channels from dominating L1.

---

### 4.3 \(L_{\mathrm{feat}}\) — whitened spatial L1 (weight 1.0)

`weighted_l1`: mean absolute error **over channels first**, then weight
space:

\[
e_{bhw} = \frac{1}{128}\sum_c \bigl|(\bar F_s - \bar F_t)_{bchw}\bigr],
\quad
L_{\mathrm{feat}} = \mathrm{mean}(w \odot e).
\]

\(F_t\) is detached. With \(w=1\) this is ordinary mean L1 on whitened maps.

---

### 4.4 \(L_{\mathrm{cos}}\) — per-location cosine (weight 0.2)

For each spatial site, treat the 128-vector as a direction:

\[
u = \frac{F_s(\cdot,h,w)}{\|F_s(\cdot,h,w)\|},\quad
v = \frac{F_t(\cdot,h,w)}{\|F_t(\cdot,h,w)\|},
\quad
\ell_{hw} = 1 - u^\top v.
\]

\[
L_{\mathrm{cos}} = \frac{\sum_{hw} w_{hw}\,\ell_{hw}}{\sum w}.
\]

Unwhitened \(F\) (scale-invariant per cell). Complements L1, which cares
about amplitude.

---

### 4.5 \(L_{\mathrm{rel}}\) — spatial Gram / TinyMIM-style (weight 0.2)

Flatten space to \(N=256\) locations. L2-normalize **across channels** so
each location is a unit 128-vector; \(P,T \in \mathbb{R}^{B\times 128\times 256}\):

\[
G_s = P^\top P,\quad G_t = T^\top T \in \mathbb{R}^{256\times 256},
\quad
L_{\mathrm{rel}} = \mathrm{MSE}(G_s, G_t).
\]

Matches pairwise spatial similarities (which cells co-activate), not
absolute features. Parent Pair-CF PCA is **not** this Gram; PCA is on
Giant token pair-diffs.

---

### 4.6 \(L_{\mathrm{mgd}}\) — masked predictor (weight 0.5)

`JEPAPredictor`: identity skip + depthwise residual + **zero-init** 1×1
(starts as identity). 2D sincos pos-emb added inside the residual.

Bernoulli mask \(M_{bhw} \sim \mathrm{Bern}(0.5)\) on the **16×16 grid**
(not tubelets):

\[
h = M \odot (F_{\mathrm{ctx}} + \mathrm{pos}),
\quad
\hat F = F_{\mathrm{ctx}} + \mathrm{Conv}_{1\times1}(\mathrm{DWBlock}(h)).
\]

Visible cells still go through the skip \(F_{\mathrm{ctx}}\). Masked cells
must be filled by the predictor. Then the same whitened weighted L1 as
\(L_{\mathrm{feat}}\), target \(\bar F_t\).

This is **not** V-JEPA’s predictor and **not** Pair-CF shrink. It is a
small CNN on student `F_ctx` only.

---

### 4.7 \(L_h\) — occupied Smooth-L1 vs GT (weight 0.8)

Peak class uses 0.25 / 0.60. Occupied cells: \(Y_{ij} \ge 0.25\).

\[
w_{ij}
= 0.12\,(1-o_{ij}) + 4.0\, o_{ij}\,(0.5 + \mathrm{clip}(Y_{ij},0,1)),
\quad o=\mathbf{1}[Y\ge 0.25].
\]

\[
L_h = \mathrm{mean}_{ij}\Bigl[
w_{ij}\cdot \mathrm{SmoothL1}_{\beta=0.05}(\hat H_{ij}, Y_{ij})
\Bigr].
\]

Empty cells still get a small pull (0.12). Occupied cells get 4×(0.5+Y)
so high-GT STOP cells weigh more. **Not** BCE; no appear×8.

---

### 4.8 \(L_{\mathrm{floor}}\) — GT-STOP peak (weight 0.6)

Only samples with \(\max(Y)\ge 0.60\):

\[
L_{\mathrm{floor}}
= \mathrm{mean}_{b:\,\mathrm{GT=STOP}} \mathrm{relu}\bigl(0.70 - \max_{ij}\hat H_b\bigr).
\]

If no STOP in the batch, 0. Target **0.70** is slightly above the 0.60
class cut (parent hinge used 0.65). Does not cap SAFE peaks (no SAFE hinge).

---

### 4.9 \(L_{\mathrm{kd}}\) — selective parent heatmap (weight 0.4)

Let \(y_g, y_p\) be peak classes of GT and parent \(H_t\). Per sample:

\[
w_b = \mathbf{1}[y_g=y_p]\cdot \bigl(1-\mathbf{1}[y_p=\mathrm{STOP}\wedge y_g\neq\mathrm{STOP}]\bigr).
\]

The extra factor is redundant with “agree” (if parent is STOP and GT is
not, they disagree). Effective: **only windows where parent class = GT
class**.

\[
L_{\mathrm{kd}}
= \frac{\sum_b w_b \|\hat H_b - H_{t,b}\|_1 }{(\sum_b w_b)\cdot 25}.
\]

Parent false STOP is not distilled. Parent-correct STOP/SAFE/THREAT maps
are.

---

### 4.10 Decoder L2 anchor (weight 0.01)

Mean over parameter tensors of \(\mathrm{mean}((\theta-\theta_0)^2)\), not
a global RMS. \(\theta_0\) is Pair-CF init. Stops the unfrozen decoder
from walking far while still allowing tiny ep9–10 steps.

---

### 4.11 Total

\[
\begin{aligned}
L_{\mathrm{latent}} &= L_{\mathrm{feat}} + 0.2 L_{\mathrm{cos}} + 0.5 L_{\mathrm{mgd}} + 0.2 L_{\mathrm{rel}}, \\
L_{\mathrm{task}} &= 0.8 L_h + 0.6 L_{\mathrm{floor}} + 0.4 L_{\mathrm{kd}}, \\
L &= L_{\mathrm{latent}} + L_{\mathrm{task}} + 0.01 L_{\mathrm{dec\_reg}}.
\end{aligned}
\]

At official epoch 10, train means: \(L_{\mathrm{feat}}\approx 0.462\),
\(L_{\mathrm{cos}}\approx 0.173\), \(L_{\mathrm{mgd}}\approx 0.463\),
\(L_{\mathrm{rel}}\approx 0.053\), \(L_h\approx 0.030\),
\(L_{\mathrm{floor}}\approx 0.016\), \(L_{\mathrm{kd}}\approx 0.028\),
\(L_{\mathrm{dec\_reg}}\approx 10^{-8}\). Weighted latent is ~0.75 of
\(L_{\mathrm{total}}\approx 0.78\); task is a small add-on.

No EMA mix, no Pair-CF PCA, no parent hinges.

---


## 5. Optimization

- AdamW: encoder + predictor lr **3e-4**, proj **1e-4**, wd 1e-4
- Separate decoder AdamW; **lr = 3e-4 × multiplier**
- Cosine schedule, 5 warmup epochs, min lr 1e-6 (student only)
- Grad clip **1.0** on all parameters
- AMP **bfloat16** on GPU; CPU eval float32
- 30 epochs, batch 8

**Decoder unfreeze schedule** (same table as M2–M8):

| Epochs | Decoder LR multiplier |
|---|---:|
| 1–8 | 0 (frozen, `requires_grad=False`) |
| 9–10 | 0.001 |
| 11–12 | 0.002 |
| 13–14 | 0.005 |
| 15–16 | 0.01 |
| 17–20 | 0.02 |
| 21–25 | 0.03 |
| 26–30 | 0.05 |

Official M1 is **epoch 10**: decoder has been unfrozen for two epochs at
multiplier 0.001 (lr \(3\times 10^{-7}\)). Measured drift vs Pair-CF:
RMS **8.2e-5**. Decoder hash at init must be `77f78c8714667617`.

---

## 6. Official selection metrics (epoch 10, selection only)

Fixed peaks 0.25 / 0.60. \(n=5120\). Confusion
`[[3119,153,322],[376,71,161],[310,47,561]]` (rows GT SAFE/THR/STOP).

| Metric | Value |
|---|---:|
| SAFE recall | **0.868** |
| THREAT recall | 0.117 |
| STOP recall | **0.611** |
| STOP precision | 0.537 |
| Balanced (SAFE+STOP)/2 | 0.739 |
| Pre-fork STOP | **0.545** |
| Post-fork STOP | 0.720 |
| GT-STOP peak p10 | 0.012 |
| Collapse (GT-STOP peak < 0.25) | 0.338 |
| `F` cosine vs parent | **0.810** |
| GT heatmap MSE | 0.035 |

Gates **SAFE ≥ 0.82 and STOP ≥ 0.70** and **SAFE ≥ 0.85 and STOP ≥ 0.60**
are **not** met. Later M2–M9 did not beat this STOP at SAFE ≥ 0.85.

THREAT recall is weak: many caution maps are predicted SAFE or STOP.

---

## 7. Why it is still “best”

Compared with parent-style students (M7/M8) and heatmap-dominant:

- **Parent cosine ~0.81** vs ~0.17 (M8) / ~0.66 (late M7). The decoder was
  trained on parent `F_dec`; matching that map keeps calibration usable.
- Decoder almost unmoved at ep10, so heatmap = “parent decoder on a
  student feature that actually looks like `F_dec`”.
- STOP floor 0.70 is weaker than the parent STOP hinge, but latent match
  prevents the SAFE=0 / all-STOP regime of early M7–M9.
- Remaining failure modes (selection audit): **amplitude** on GT-STOP
  (collapse 0.34, p10 0.012), **pre-fork** STOP 0.55, **RIGHT** weaker
  than LEFT/CENTER. Many misses are same-column, not a full spatial miss.

Unfreezing the decoder **further** (ep14–30) raises SAFE and cosine
slightly and **drops** STOP (ep30 STOP 0.52). So the official point is
early in the unfreeze schedule, not “more decoder FT”.

---

## 8. Code map

| Piece | Location |
|---|---|
| Encoder | `student_model/encoder.py` (`StudentEncoder`) |
| Overnight wrap + 1×1 | `collision_jepa3/student_research/overnight6/models.py` |
| Losses | `collision_jepa3/student_research/overnight6/losses.py` (`mid == 1`) |
| Train loop | `collision_jepa3/student_research/overnight6/train.py` |
| Launch | `scripts/81_train_overnight6.py --model 1` |
| Decoder schedule | `collision_jepa3/student_research/overnight6/schedule.py` |
| Metrics | `scripts/75_train_student_lat_paircf.py` (`metrics_from_rows`) |
