# Model trial 4

Now-only collision teacher. A frozen V-JEPA 2 ViT-G encodes the clip. The trunk below is the only trained network. It emits `Hhat_now` and `Hhat_now_logits`, both `[B, 5, 5]`. There is no `Z_t`, no future-horizon head, and no LoRA phase.

The dataset, pairing, and anchor rules match trial 3, including the three future heatmap fields. The loss does not read those fields. Training uses the trial 3 now-map BCE-with-logits (`L_now`, occupied-cell pos weight 2). Row weights apply only if `loss.heatmap` is set to `mse`.

Encoder tokens are reused from the trial 3 cache (`paths.cache_root`). Checkpoints and logs stay under this directory.

## Trunk

`[B, 2048, 1408]` post-encoder tokens, already LayerNormed, ordered `(T=8, H=16, W=16)`.

1. Linear `1408→128` and GELU. Channels move to `[B, 128, 8, 16, 16]`.
2. Two spatiotemporal blocks, independent weights. Each block runs depthwise `(5,3,3)` and `(3,5,5)` branches (`bias=False`), per-location LayerNorm and GELU, concatenates to 256 channels, pointwise `256→128`, LayerNorm, and adds the input.
3. Conv3d `128→96`, kernel `(3,1,1)`, stride `(2,1,1)` → `[B, 96, 4, 16, 16]`.
4. Path A collapses time with Conv3d kernel `(4,1,1)` to 48 channels. Path B keeps the last of the four slices (96 channels). A `1×1` conv mixes 144 channels down to 48.
5. Conv2d `48→16`, `3×3`, ReLU, then Conv2d `16→1` logits at `16×16`.
6. Adaptive max-pool to `5×5`, then sigmoid.

The trunk has 347,745 parameters when the encoder width is 1408.

## Caveats

- LayerNorm is over the 128 channels at each location. Applying `nn.LayerNorm(128)` on `[B, 128, T, H, W]` would normalize the width axis instead.
- The “latest” slice is the last output of the stride-2 temporal conv. With padding 1 that window covers tubelets 5, 6, and 7 and is centered on tubelet 6. It is not the last RGB frame.
- Adaptive max-pool from 16 to 5 uses overlapping windows `[0:4), [3:7), [6:10), [9:13), [12:16)`. A fine cell on a boundary can raise two coarse cells.

## Smoke test

From `model_trial_4/`:

```bash
python scripts/03_smoke_test.py
```

That builds a dummy encoder with the ViT-G token layout (`2048×1408`), runs one BCE step, and checks that backbone weights stay frozen. It does not download V-JEPA.

Training still needs a manifest (`scripts/01_build_manifest.py`) and the shared token cache. `scripts/04_train_teacher.py` is frozen-only.
