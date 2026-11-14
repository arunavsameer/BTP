# CollisionStudentV2 + Canny (trial 6)

Tiny RGB student (~0.4M) that copies the teacher’s `F_dec` then reuses the **frozen** Pair-CF heatmap decoder. Canny is a preprocess only (not in the weights). Inspired by the CollisionStudentV2 walkthrough; not a dump of the 1s teacher loss kitchen sink.

Caches are **read-only** (not copied into this folder):

- RGB 256 letterbox: `/storage/BTP/model_trial_5/shared_rgb_cache_256_v1`
- Teacher `F_dec` / `H_teacher`: `/storage/BTP/model_trial_3/checkpoints/parent_target_cache_v2`
- Decoder weights: `pair_cf_treatment_s1337_best.pt` (hash `77f78c8714667617`)
- Optional channel stats: `/storage/BTP/model_trial_5/student_runs/fdec_channel_stats.npz`

## Canny

Order **(a)** from `model_trial_3/docs/student_edge_street_vlog.md`: the cache is already 256 letterbox, then per-frame

1. `cv2.GaussianBlur` 5×5, `sigma=0` (OpenCV sets σ≈1.1)
2. `cv2.Canny` low **40**, high **120**
3. Paint mask pixels **cyan** `(0, 255, 255)` onto a copy of the RGB photo

The clip stays **3-channel** `[B,16,3,256,256]`. No 4th edge channel — the stem is `Conv3D 3→16`, and cyan-on-photo is what the street-vlog student already consumed. ImageNet mean/std, then permute to `[B,3,16,256,256]` for Conv3D.

## Forward (batch=1)

| Stage | What | Shape |
|---|---|---|
| Input | 16 frames, 6-frame stride, 256 letterbox | `[1,3,16,256,256]` |
| Stem 1a | Conv3D 3→16, k=3³, stride 2³, pad 1, GN+GELU | `[1,16,8,128,128]` |
| Stem 1b | InvRes2p1d 16→32, spatial DW stride 2, **causal** temporal DW k=3 | `[1,32,8,64,64]` |
| Stem 1c | same 32→64 | `[1,64,8,32,32]` |
| Stem 1d | 64→96 via expand **128** then project, spatial down | `Fstem [1,96,8,16,16]` |
| Motion | Δt = Ft−Ft−1 (Δ0=0), cat 192, 1×1 192→128 GN+GELU, CausalInvResSE d=1 then d=2 | `Fmotion [1,128,8,16,16]` |
| Readout | last slice 128→64; RecencyAttnPool over T=8 (1×1 scores + learned bias init `linspace(-1.4,0)`) then 128→64; cat | `Fctx [1,128,16,16]` |
| Predictor | 256 tokens + 2D sincos PE, 2 Bottleneck MHSA (QKV 128→192→64, 4×16, FFN 128→155→128), unflatten, 2 LayerScaleDW (DW3×3, GN, GELU, PW, SE, γ=0.001) | `Fpred [1,128,16,16]` |
| Decoder | frozen Pair-CF: 128→48 1×1, 3×3, dilated 3×3 d=2, 48→1, AdaptiveMaxPool 16→5, sigmoid | `Ĥ [1,5,5]` |

SAFE if peak `< 0.25`, STOP if peak `≥ 0.60`. Decoder is never trained; heatmap loss still backprops **through** it into the student.

Causal time convs left-pad only (`t-2,t-1,t` at dilation 1; `t-4,t-2,t` at dilation 2) so the student cannot peek at future slices — a small change vs a symmetric pad.

## Loss vs teacher

Walkthrough: `L = Lfocal(H, P_teacher_5) + LBCE(Z16, Z_teacher_16) + 0.5 (Lcosine + 0.5 SmoothL1)` on `Fpred`.

What we actually use:

| Term | Weight | Target |
|---|---|---|
| Focal 5×5 (hazard-weighted, γ=2) | 1.0 | `H_teacher` (cached 5×5) — **primary heatmap** |
| `Lcosine + 0.5 SmoothL1` | 0.5 | `F_dec` `[128,16,16]` |
| Whitened L1 | 0.25 | same `F_dec`, using trial-5 `mu`/`sigma` (read-only) |
| STOP peak hinge ≥0.60 | 0.25 | `gt_1s` (trial 5/3, cheap) |
| SAFE peak hinge `<0.25` | 0.15 | `gt_1s` |

**Omitted:** Z16 BCE — `parent_target_cache_v2` has `Z_t.npy` as the **32×5×5 adapter bottleneck**, not decoder 16×16 logits. Gram / whitening-as-a-whole / blob / direction / Pair-CF PCA are not copied.

## Train / not train

`scripts/train_cnn_edge.py` + `./run_cnn_edge_local.sh` (batch 4, accum 8 on a 4060). Refuses to start if Atto (`90_train_convnextv2_student.py`) is still running. This folder is code-only until that job finishes.
