# Model trial 4

1s-ahead collision teacher. A frozen V-JEPA 2 ViT-G encodes the 16-frame clip that ends at anchor `t`. The trunk below plus a one-parameter consistency mixer are the trained network. The mixer emits `Hhat_1s` and `Hhat_1s_logits`, both `[B, 5, 5]`, trained against the heatmap at `t+30` (`h_1s`). There is no `Z_t`, no now-map head, no later-horizon head, and no LoRA phase.

The dataset, pairing, and anchor rules match trial 3. `h_now`, `h_1_5s`, and `h_2s` stay in the batch; the loss and checkpoint selector only read `h_1s`. Training uses BCE-with-logits (`L_1s`, occupied-cell pos weight 2). Ambiguous pre-fork 1s maps use the collision/miss mean (`sampling.ambiguous_policy: mean_target`). Row weights apply only if `loss.heatmap` is set to `mse`.

Encoder tokens are reused from the trial 3 cache (`paths.cache_root`). Checkpoints and logs stay under this directory.

## Trunk

`[B, 2048, 1408]` post-encoder tokens, already LayerNormed, ordered `(T=8, H=16, W=16)`.

1. Linear `1408→128` and GELU. Channels move to `[B, 128, 8, 16, 16]`.
2. Two spatiotemporal blocks, independent weights. Each block runs depthwise `(5,3,3)` and `(3,5,5)` branches (`bias=False`), per-location LayerNorm and GELU, concatenates to 256 channels, pointwise `256→128`, LayerNorm, and adds the input.
3. Conv3d `128→96`, kernel `(3,1,1)`, stride `(2,1,1)`, right-pad 1 on time → `[B, 96, 4, 16, 16]`. The last slice is centered on tubelet 7 (RGB `t-6` and `t`); tubelet 6 still sits in that kernel.
4. Path A collapses time with Conv3d kernel `(4,1,1)` to 48 channels. Path B keeps that last slice (96 channels). A `1×1` conv mixes 144 channels down to 48.
5. On the 16×16 map: Conv2d `48→48` `3×3`, ReLU, dilated Conv2d `48→48` `3×3` dilation 2, ReLU, then `1×1` `48→16` and `1×1` `16→1` logits.
6. Adaptive max-pool to `5×5`, then sigmoid.

The trunk has 383,265 parameters when the encoder width is 1408.

## Consistency mixer

The trunk still emits a raw 5×5 map. A one-parameter mixer then writes the map that the loss and the vest see:

`alpha = exp(-std / tau)`, `final = alpha * raw + (1 - alpha) * previous_final`.

`std` is occupancy-weighted per-cell spread over the last `smoother.window` (default 4) **raw** maps of the same clip. It does not use `|raw - final|`. The first snapshot of a clip copies raw. `tau` starts at `smoother.tau_init: 0.5` so `alpha` begins high; training learns `log tau` with the trunk. Mix and logit stay in fp32 (AMP bfloat16 cannot represent `1 - 1e-4`, so `logit(1)` was Inf). Clip-contiguous sampling is required so a batch is time order inside a clip. State resets on a new `clip_id`, at each epoch, and at the start of eval. `Hhat_1s` / `Hhat_1s_logits` are the mixed map; `Hhat_1s_raw` is the trunk output.

## Caveats

- LayerNorm is over the 128 channels at each location. Applying `nn.LayerNorm(128)` on `[B, 128, T, H, W]` would normalize the width axis instead.
- The “latest” slice is the last output of the stride-2 temporal conv. Right-pad 1 makes that window tubelets `(6, 7, pad)`, centered on the last RGB tubelet. Symmetric pad would have been `(5, 6, 7)` centered on 6.
- Adaptive max-pool from 16 to 5 uses overlapping windows `[0:4), [3:7), [6:10), [9:13), [12:16)`. A fine cell on a boundary can raise two coarse cells.

## Smoke test

From `model_trial_4/`:

```bash
python scripts/03_smoke_test.py
```

That builds a dummy encoder with the ViT-G token layout (`2048×1408`), runs one BCE step, and checks that backbone weights stay frozen. It does not download V-JEPA.

Training still needs a manifest (`scripts/01_build_manifest.py`) and the shared token cache. `scripts/04_train_teacher.py` is frozen-only.
