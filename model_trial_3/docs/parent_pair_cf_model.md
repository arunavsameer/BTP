# Pair-CF parent (collision teacher)

This is the **parent** the RGB students copy: checkpoint
`model_trial_3/checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt`,
seed **1337**, decoder state hash **`77f78c8714667617`**.

It predicts a **1-second-ahead 5×5 occupancy heatmap** from **16 RGB frames**.
It is not a V-JEPA self-supervised run: V-JEPA Giant is a **frozen encoder**.
What is trained is a spatial–temporal adapter, a heatmap decoder, and a
pair-counterfactual regularizer on tokens.

---

## 1. Task

**Input.** Sixteen frames ending at anchor \(t\), offsets

`[-90, -84, …, -6, 0]`

(3 s of history at 6-frame / 0.2 s spacing, 30 fps). Frames are **letterboxed**
to 256×256 (FOV preserved) and ImageNet-normalized.

**Output.** A 5×5 map \(\hat H_{1s} \in [0,1]^{5\times5}\): occupancy **one
second after** \(t\) (frame \(t+30\)).

**Warning class** from peak \(\max(\hat H)\):

| class | peak |
|---|---|
| SAFE | \(< 0.25\) |
| THREAT (caution) | \(0.25 \le \text{peak} < 0.60\) |
| STOP | \(\ge 0.60\) |

**Fork.** Scenario fork is frame **365**. Pre-fork history is shared between
collision and miss of the same sample; futures diverge.

**Ambiguous futures.** When bg-pair futures disagree, targets use the
**cell-wise mean** of collision and miss (`ambiguous_policy: mean_target`).
Ambiguous 1 s maps can be masked (`amb_1s`).

**Grid.** Rows are depth (near→far), columns LEFT `[0,1]`, CENTER `[2]`,
RIGHT `[3,4]`.

---

## 2. Architecture

```
16× RGB 256  →  frozen V-JEPA2 Giant  →  tokens
                 SpatialTemporalAdapter
                      ├── F_dec  [B,128,16,16]   ← heatmap is read here
                      └── Z_t    [B, 32, 5, 5]   ← close / evidence / pair-CF
                 HeatmapDecoder (dilated, max-pool)
                      └── Hhat_1s, logits  [B,5,5]
```

### 2.1 Frozen V-JEPA Giant

- Hugging Face id: `facebook/vjepa2-vitg-fpc64-256`
- `tuning_mode: frozen`, `skip_predictor: true` (no JEPA predictor loss)
- `torch_dtype` bfloat16 in train AMP
- Tokens are typically **cached** (`cache.cache_encoder_tokens`); RGB is not
  re-encoded every step
- Backbone stays `eval()` even when the teacher is in train mode

Giant is **not** updated in Pair-CF treatment. Phase flag `--phase frozen`.

### 2.2 Spatial–temporal adapter (trainable)

Tokens reshape to tubelet grid \(T'\times 16\times 16\). Then:

1. Per-tubelet **spatial proj**: `1×1 → GELU → 3×3` to **128** channels
2. Four **temporal residual** blocks, dilations **1, 2, 1, 2** (depthwise
   time conv + pointwise + GroupNorm + GELU)
3. **Softmax pool over time** with a recency bias → `[B,128,16,16]`
   This map is **`F_dec` / `fine`**: what the decoder (and students) use
4. Adaptive avg-pool to 5×5 and a bottleneck `128 → 32` + LayerNorm → **`Z_t`**

`decode_from: fine`: heatmaps are decoded from the 128-ch 16×16 map (optionally
resized to `heatmap_grid: 16`), **not** from \(Z_t\).

Optional **task-whiten drop** (drop 16 channels) can sit on `fine` before the
decoder; Pair-CF still reads this path.

### 2.3 Heatmap decoder (trainable, copied onto students)

`HeatmapDecoder` on 128-ch maps:

- `1×1` 128→48, ReLU
- `3×3` 48→48, ReLU
- **dilated** `3×3` (dilation 2), ReLU
- `1×1` 48→1 logits at 16×16
- **adaptive max-pool** to 5×5, then **sigmoid**

Sigmoid-after-max-pool equals max-of-sigmoids, so 5×5 logits are valid for
BCE-with-logits. No BatchNorm.

Hash of this decoder in the best Pair-CF ckpt: **`77f78c8714667617`**.

### 2.4 Extra heads (parent-only)

On **`Z_t`**, not on `F_dec`:

- **close head** (`1×1`): approaching occupancy aux (annotations)
- **evidence head**: unused in this run (`beta_nll_weight: 0`)
- **no** 1.5 s / 2 s residual predictors (`heads.mode: direct_h1s`)

`direct_h1s` means the decoder output **is** \(H_{1s}\). There is no \(H_{now}\)
head and no copy-residual \(\Delta\).

---

## 3. Loss (treatment s1337)

Implemented in `collision_jepa3/losses/composite.py` (`TeacherLoss`).
Let \(\ell \in \mathbb{R}^{B\times 5\times 5}\) be **decoder logits**,
\(\hat H = \sigma(\ell)\) after 5×5 max-pool, \(Y = h_{1s}\) the (possibly
group-mean) GT. Sample mask \(m_b = \mathbf{1}[\neg \mathrm{amb}_{1s}]\).
Heatmap reconstruction is **BCE on logits** (`loss.heatmap: bce`), not MSE.

Total (before Pair-CF aux):

\[
\begin{aligned}
L_{\mathrm{task}}
&= L_{1s}
 + 0.35\,L_{\mathrm{blob}}
 + 1.0\,L_{\mathrm{stop}}
 + 0.5\,L_{\mathrm{safe}}
 + 1.0\,L_{\mathrm{close}}
 + 0.10\,L_{\mathrm{FP}}
 + 0.05\,L_{\mathrm{dir}}
 + 0.02\,L_{\mathrm{bg}}.
\end{aligned}
\]

Hinges and \(L_{\mathrm{FP}}\), \(L_{\mathrm{dir}}\) use **probabilities**
\(\hat H\) (`logit_hinges: false`). \(L_{1s}\) and \(L_{\mathrm{blob}}\) use
**logits**.

**Off:** 1.5 s / 2 s / \(H_{\mathrm{now}}\) / \(\Delta\), VICReg, beta-NLL,
prob-STOP, student latent L1.

---

### 3.1 Cell-wise BCE \(L_{1s}\) (`soft_bce_logits`)

Per cell:

\[
\mathrm{BCE}_{bij}
= \mathrm{binary\_cross\_entropy\_with\_logits}(\ell_{bij},\, Y_{bij}),
\quad Y \in [0,1].
\]

Weights (multiplied, then a **global** weighted mean):

\[
w_{bij}
= \underbrace{\mathbf{1}[Y_{bij}>0.1]\cdot 2 + \mathbf{1}[Y_{bij}\le 0.1]}_{pos\_weight}
\cdot w^{\mathrm{appear}}_{bij}
\cdot w^{\mathrm{small}}_{bij}
\cdot m_b.
\]

**Appear** (`bce_appear_weight = 8`, needs \(H_{\mathrm{now}}\)):

\[
w^{\mathrm{appear}}_{bij}
=
\begin{cases}
8 & H^{\mathrm{now}}_{bij} < 0.1 \;\wedge\; Y_{bij} \ge 0.5 \\
1 & \text{otherwise}.
\end{cases}
\]

Empty-now → occupied-future cells (jaywalker / approach) dominate the
gradient. Disappear weight is 1.0 in this run (off).

**Small STOP blob** (`small_blob_weight = 2`, `max_cells = 4`): if
\(\max(Y)\ge 0.60\) and the number of cells with \(Y\ge 0.5\) is in
\(\{1,\ldots,4\}\), occupied cells get ×2. Large blobs stay at 1. This stops
a 2-cell pre-fork STOP from being drowned by a 7-cell post-fork map in the
cell average.

\[
L_{1s} = \frac{\sum_{bij} w_{bij}\,\mathrm{BCE}_{bij}}{\sum_{bij} w_{bij}+\varepsilon}.
\]

---

### 3.2 Blob BCE \(L_{\mathrm{blob}}\) (`blob_bce_logits`)

3×3 max-pool (stride 1, pad 1) on logits and on GT:

\[
\tilde\ell_{ij} = \max_{|i'-i|\le 1,\,|j'-j|\le 1} \ell_{i'j'},
\quad
\tilde Y_{ij} = \max_{\mathrm{nbhd}} Y.
\]

Keep only neighbourhoods with \(\tilde Y \ge 0.5\) (and valid \(m_b\)).
Then mean BCE-with-logits on those cells. Lighting the true cell **or a
neighbour** satisfies the term. Empty 3×3s are excluded so it cannot invent
blobs on empty maps.

---

### 3.3 Peak hinges (probability space)

Let \(p_b = \max_{ij} \hat H_{bij}\), \(g_b = \max_{ij} Y_{bij}\).

**STOP hinge** (weight 1.0): only GT-STOP windows (\(g_b \ge 0.60\)):

\[
L_{\mathrm{stop}}
= \frac{1}{|S|}\sum_{b\in S} \mathrm{relu}(0.65 - p_b),
\quad S=\{b: g_b\ge 0.60,\, m_b=1\}.
\]

Default target is `stop_threshold + margin` = \(0.60+0.05=0.65\). SAFE/THREAT
windows do not enter.

**SAFE hinge** (weight 0.5): only GT-SAFE (\(g_b < 0.25\)):

\[
L_{\mathrm{safe}}
= \frac{1}{|A|}\sum_{b\in A} \mathrm{relu}(p_b - 0.25).
\]

True STOP windows are excluded so STOP recall is not pulled down.

---

### 3.4 False-positive \(L_{\mathrm{FP}}\)

On cells with \(Y_{ij} < 0.05\) (and valid sample):

\[
L_{\mathrm{FP}}
= \mathrm{mean}\bigl(\mathrm{relu}(\hat H_{ij} - 0.10)^2 \bigr)
\quad\text{over those cells}.
\]

Margin `false_positive_margin = 0.10`. Empty cells may sit up to 0.10 for
free; above that they are squared.

---

### 3.5 Direction \(L_{\mathrm{dir}}\)

Row weights are uniform \([1,1,1,1,1]\). For a map \(H\), score each side
with a **soft max** (log-sum-exp, temperature \(T=0.08\)):

\[
s_{\mathrm{LEFT}}(H)
= T\log\sum_{i,\,j\in\{0,1\}} \exp(H_{ij}/T),
\]

and similarly CENTER \(j=2\), RIGHT \(j\in\{3,4\}\). Stack
\(s(H)\in\mathbb{R}^3\).

Apply only if the **GT** side-score max \(\ge 0.25\) (CAUTION+):

\[
L_{\mathrm{dir}}
= \mathrm{SmoothL1}\bigl(s(\hat H),\, s(Y)\bigr)
\quad\text{on those samples}.
\]

---

### 3.6 Close head \(L_{\mathrm{close}}\)

`Hhat_close = \sigma(\mathrm{Conv}_{1\times1}(Z_t))` vs annotation closing
maps. Soft BCE, pos_weight 2, only on occupied aux cells. Weight 1.0 in
treatment. Students do not use this head.

---

### 3.7 Background consistency \(L_{\mathrm{bg}}\)

When a background-pair mate \(Z^{(b)}\) is in the buffer (and GT 1 s maps
agree within tol):

\[
L_{\mathrm{bg}}
= \mathrm{mean}_{ij}\bigl(1 - \cos(Z^{(a)}_{:,ij},\, Z^{(b)}_{:,ij})\bigr).
\]

Weight 0.02. This is **not** the PCA shrink; it is a cosine pull on \(Z_t\).

---

### 3.8 Pair-difference PCA (the “Pair-CF” logic)

Fit is **offline** (`scripts/23_pair_counterfactual_suite.py`,
`scripts/pair_cf_lib.py`), saved as
`reports/pair_background_counterfactual/subspace/learned_k16.npz`.
Training only **loads \(U\)** and applies a shrink.

#### Fit (once, on train pairs)

Tokens per window: \([T=2048,\, C=1408]\) (Giant). Pool to a clip vector

\[
F = \mathrm{mean}_{t=1}^{T} \,\mathrm{token}_t \in \mathbb{R}^{1408}.
\]

For each valid background pair \((a,b)\) at the **same anchor** (matched
collision/miss or two bgs of one sample):

\[
D = \tfrac12 (F_a - F_b) \in \mathbb{R}^{1408}.
\]

Stack train \(D\) rows. Center \(\mu_D = \mathrm{mean}(D)\), then
sklearn **PCA** (`n_components=16`, randomized SVD):

\[
X = D - \mu_D,
\quad
U \in \mathbb{R}^{16\times 1408},\quad U U^\top \approx I_{16}.
\]

\(U\) are the top-16 principal **channel** directions of *pair difference*,
i.e. “what typically changes when only background / outcome pairing
changes.” Also stored: \(\mu_F\), eigenvalues, explained-variance ratios.
\(k\in\{4,8,16\}\) were fit; treatment uses **k=16**.

Intent: those directions are mostly **nuisance** (lighting/bg identity).
They are not assumed to be occupancy; occupancy should survive projection
*off* this subspace.

#### Intervention at train time (`apply_pair_counterfactual_shrink`)

Paired batch is laid out as first half clip \(A\), second half clip \(B\)
(same \(B\) size). Token tensors \(F_A, F_B \in \mathbb{R}^{B/2 \times T \times C}\):

\[
\bar F_A = \mathrm{mean}_T F_A,\quad
\bar F_B = \mathrm{mean}_T F_B,\quad
D = \tfrac12(\bar F_A - \bar F_B).
\]

Project \(D\) onto the PCA subspace and scale by \(\lambda=1\):

\[
d = \lambda \,(D U^\top)\, U \in \mathbb{R}^{C}
\qquad\text{(broadcast over tokens)}.
\]

Counterfactual tokens:

\[
\tilde F_A = F_A - d,\qquad \tilde F_B = F_B + d.
\]

This **halves** the pair-difference *inside* \(\mathrm{span}(U)\) and leaves
the orthogonal complement (hoped occupancy) untouched. Correction is
**channel-global**: the same \(d\) is added to every token position
(tubelet).

#### When it fires, and how it enters the loss

- Only if `bg_reg_paired` (true pair batch).
- Hash of `(epoch, pair_id, "pair_cf_aux")` vs `pair_frac=0.5` → ~50% of
  pair batches.
- Strength ramp (same as clean-anchor): epoch 0 off, then 0.25, 0.50,
  0.75, then 1.0.
- \(\beta_{\mathrm{cf}}\) is calibrated once:
  \(\beta = 0.10 / \mathrm{median}(\|g_{\mathrm{cf}}\| / \|g_{\mathrm{clean}}\|)\)
  so the aux gradient is about **10%** of the clean task gradient
  (`pair_cf_grad_scale=0.10`).

Forward the **shrunk** tokens through adapter+decoder, compute the **same**
\(L_{\mathrm{task}}\) on GT, then

\[
L = L_{\mathrm{task}}^{\mathrm{clean}} + \beta_{\mathrm{cf}}\cdot\mathrm{ramp}\cdot L_{\mathrm{task}}^{\mathrm{CF}}.
\]

The clean path is untouched. The aux path trains the adapter to still
explain occupancy after background-ish channels are cancelled.

This shrink is **not** used for students. Students never see Giant tokens.

---


## 4. Data and training

- Dataset: Blender groups (`groups15_n400_s67_n400`) plus extra same-dir
  samples 207–229 on **train only**
- Anchors: **group_random** — 26 pre-fork + 14 post-fork timestamps shared
  inside a scenario group
- Optimizer: AdamW, adapter lr **3e-4**, cosine, 5 warmup epochs, grad clip 1.0
- AMP bfloat16
- Early stop: `balanced_warning` with SAFE floor 0.82, STOP floor 0.55,
  post-fork STOP floor 0.88, patience 6 after epoch 8
- Max steps 28350 (frozen phase)

Student caches later reuse **group-mean GT** and parent `F_dec` / `H_teacher`
on the same 16-offset RGB windows. Splits used for students:
**22 680 train / 5 120 selection / 5 120 test**. Test is never used for
student training or checkpoint picking.

---

## 5. What students are allowed to copy

| Tensor | Shape | Use |
|---|---|---|
| `F_dec` | `[128,16,16]` | Distillation target (M1 latent losses) |
| `H_teacher` | `[5,5]` | Selective KD when parent class agrees with GT |
| decoder weights | 47 809 params | Init; hash must stay `77f78c8714667617` at start |
| GT `h_1s` | `[5,5]` | Occupancy / parent-style BCE |

Students **do not** run Giant, do not train Pair-CF shrink, and do not use
\(Z_t\) or the close head.

---

## 6. Code map

| Piece | Location |
|---|---|
| Teacher module | `collision_jepa3/models/teacher.py` |
| Adapter | `collision_jepa3/models/spatial_adapter.py` |
| Decoder | `collision_jepa3/models/heatmap_decoder.py` |
| Composite loss | `collision_jepa3/losses/composite.py` |
| Pair-CF shrink (train) | `collision_jepa3/training/bg_regularization.py` (`apply_pair_counterfactual_shrink`) |
| Pair-CF PCA fit | `scripts/23_pair_counterfactual_suite.py`, `scripts/pair_cf_lib.py` (`fit_pair_pca`) |
| Aux mix in trainer | `collision_jepa3/training/trainer.py` (`_forward_loss`) |
| Launch | `submit_pair_cf_treatment_s1337.sh` |
| Defaults | `configs/default.yaml` |
