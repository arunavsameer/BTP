# model_trial_2 — V-JEPA 2 teacher → collision latent → wearable student

No region shuffle. The **teacher** is Meta’s **V-JEPA 2 Giant**
(`facebook/vjepa2-vitg-fpc64-256`), fine-tuned so its tokens decode to a **5×5
threat heatmap 1 s ahead**. Red means obstacle (minimize **miss**). Not-red on
empty road (minimize **nuisance**). After the teacher is good, a ~2M TinyCNN
**student** distills its collision latent `Z`. Only the student is for the vest.

This is the next stage of [`collision_jepa`](../collision_jepa) (frozen ViT-L +
adapter) and unlike [`model_trial_1`](../model_trial_1) (EMA of TinyCNN + shuffle).

## Architecture

```
Teacher (offline)
  unshuffled 8×256 RGB clip ending at t
       │  V-JEPA 2 Giant (frozen, then LoRA on last 6 blocks)
       ▼
  tubelet tokens
       │  SpatialAdapter (CNN → 5×5)
       ▼
  Z_t ∈ R^{32×5×5}     ← collision-relevant latent
       ├─ Decoder      → Ĥ(t)     (current threat)
       └─ Residual P   → Ẑ+
            └─ Δ head  → Ĥ(t+1s) = clamp(Ĥ(t).detach() + Δ)

Student (wearable)
  5×160 RGB  (t−20 … t)
       │  TinyCNN + loom diffs  (no shuffle)
       ▼
  Z_t → P → Ẑ+   match stopgrad(teacher Z at t+1s)
       └─ frozen teacher decoder/Δ → Ĥ(t+1s)
```

**Why fine-tune (LoRA), not freeze-only.** A frozen V-JEPA latent is generic
video. Collision “red / not red” lives in our labels. Phase 1 learns the probe.
Phase 2 lets the last ViT blocks move slightly so tokens become threat-shaped,
without training 1B weights on 600 episodes.

**If Giant will not fit 8 GB**, set `teacher.hf_model_id` to
`facebook/vjepa2-vitl-fpc64-256` (same code path, 300M).

## Pipeline

```bash
cd /storage/BTP/model_trial_2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # plus the CUDA torch you already use

python scripts/00_prepare_data.py --reset-split
python scripts/07_make_test_split.py
python scripts/01_eval_copy_baseline.py

# Stage A — teacher (downloads V-JEPA 2 into ./hf_cache)
python scripts/02_train_teacher.py

# Stage B — cache Z for every episode
python scripts/03_cache_teacher_z.py

# Stage C — student
python scripts/04_train_student.py

python scripts/05_eval.py --split both --who both
python scripts/06_overlay.py --split test --limit 8 --latch
```

Checkpoints: `checkpoints/teacher_best.pt`, `student_best.pt`.  
Selection: latched wearable score, **eligible** only if nuisance ≤ 0.20 and miss ≤ 0.25.

## Data

Same 600 native 5×5 episodes as trial 1 (mixed 250 + side 50 + pack 300). Cache
lives only under `model_trial_2/cache` (160px student + 256px teacher).
