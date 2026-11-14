# ConvNeXt-V2 Atto on RTX 4060 laptop (8 GB)

Live run (PID 105207) is **left running**. Do not change its CLI, do not start a second CUDA job, do not edit trial 3/4. This note is for the **next** run only.

Measured on the live job (2026-11-14, `--batch 4 --accum 8 --eval-every 2 --no-grad-checkpoint --num-workers 2 --eval-test`, AMP bf16):

- 22 680 train / 5 120 sel / 5 120 test windows; 5 670 train batches/epoch at batch 4.
- Epoch 1 train ≈ **22.2 min** (04:21:15 → ~04:43); first sel eval + three ~50 MB ckpts finished before 04:45, so eval is ~1.5–2.5 min not ~20 min.
- Steady train: GPU **95–100%**, 2483 MiB / 8188 MiB, 76 °C, ~56 W, SM 1920 MHz. One CUDA pid: 105207 (2442 MiB) + kwin ~7 MiB.
- `gpu_gb=2.34` (`max_memory_allocated`) is flat. Workers=2 are feeding; this is **GPU-bound**, not dataloader-idle.
- `cudnn.benchmark` is already on in `scripts/90_train_convnextv2_student.py`. `pin_memory` is on for **train only**. No `persistent_workers`, no TF32, no `torch.compile`.
- Earlier **solo** batch-8 probe: ep1 batches 1–3 at **3.53 GB allocated, no OOM**. Batch 8 OOMed only with a second trainer on the GPU.

Projected wall clock at current settings: **~11.3–11.8 h** (30 × 22.2 min train + 16 sel evals + one `--eval-test`). The ~10 h figure was slightly optimistic.

---

## 1) Cannot apply without restart

These are process lifetime / DataLoader / graph choices. The live interpreter already has them; editing the `.py` will not affect PID 105207.

| Knob | Why a restart |
|---|---|
| `--batch` / `--accum` | Activation size, cudnn algo, scheduler `steps_per_epoch` |
| `torch.compile` | Graphs the module at first steps |
| `--num-workers`, `persistent_workers`, `prefetch_factor` | Built in `_make_loaders` once |
| `--eval-every` | Read at start; also **always evals `start_ep` and last epoch** |
| `--no-grad-checkpoint`, AMP dtype, `cudnn.benchmark`, TF32 | Model / backend flags at init |
| Train-loop logging / `float(term.detach())` every batch | Already executing |

Leave PID 105207 on `--batch 4 --accum 8 --eval-every 2`.

---

## 2) Recommended next-run config (solo GPU, 8 GB with margin)

Same effective batch **32** (8×4 vs 4×8) → same 709 AdamW steps/epoch, same `--lr 3e-4`.

```bash
# from model_trial_5, after the live job exits on its own
./detach_atto.sh 8 4
# then the trainer CLI should be (run_convnextv2_atto.sh already has batch 8 accum 4):
#   --batch 8 --accum 4 --epochs 30 --lr 3e-4 --variant atto
#   --eval-every 5          # or 10; still evals epoch 1 and 30
#   --no-grad-checkpoint    # keep; VRAM is not the limiter
#   --num-workers 2         # 4 only if GPU util dips below ~85%
#   --eval-test --device cuda
```

Code deltas in `_make_loaders` for that next launch (not the live process):

- `persistent_workers=True` and `prefetch_factor=2` when `num_workers>0` (trial-3/5 teacher loaders already do this; this script does not). Workers currently **die at each epoch** (epoch-1 pids 112821/112822 were gone by ep2) and re-mmap RGB/`F_dec`/`h_now`.
- `pin_memory=True` on **sel/test** loaders too (train already pins; `collect()` uses `non_blocking=True`).
- After CUDA init, with benchmark (already set):

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

VRAM: batch 8 should land ~3.5–4.5 GB nvidia-smi of 8 GB. **Do not** stack a second job. Display compositor is 7 MiB; not the risk.

`torch.compile`: **not** in the default next CLI. `sparse_prob=0.5` is a Python branch; `apply_submanifold` is `x * mask` (full convs, not a speed win). Compile can still fuse LayerNorm3d/GRN, but first-epoch compile + extra 8 GB scratch is the wrong first knob. If tried later: `torch.compile(model.backbone, mode="default", fullgraph=False)` only after a clean batch-8 baseline, and only while this is the sole CUDA process.

---

## 3) Estimated time save vs current ~10–11.5 h

| Change | Effect | Saving |
|---|---|---|
| `--batch 8 --accum 4` | Fewer Python steps (2835 vs 5670), better Ada occupancy; live power is only ~56 W so batch 4 is not filling the 4060 | **~15–30% train** → ~18 min/epoch vs 22.2 → **~1.5–2.5 h** |
| `--eval-every 5` | Today: eval on 1,2,4,…,30 = **16** sel passes (`ep==start_ep` plus every 2). Every 5 → 7 passes (1,5,…,30). `--eval-test` stays once at the end on `checkpoint_best.pt` | **~15–25 min** (every 10: ~20–30 min) |
| `persistent_workers` | Drop 29 epoch-boundary respawns + cold mmap | **~3–8 min** |
| TF32 | Convs already bf16; leftover fp32 (custom LN, gram `bmm` in `relation_loss`) | **0–3%** |
| `torch.compile` (optional, later) | Maybe 10–20% if it stays correct | **~1 h**, not banked |

**Best guess for the recommended next run: ~8.5–9.5 h** vs measured ~11.5 h (~2–3 h saved; ~1.5–2 h vs the ~10 h estimate). Do not expect 2× from batch 8: the 3D Atto net is already at 99% util.

---

## 4) What not to do

- **Do not** start a second GPU job. Batch 8 was OOM only then. `GPU_LOCK.md` owns the 4060.
- **Do not** skip clips. This RGB cache kept all 22 680/5 120/5 120; an older trial-3 frames path dropped 920.
- **Do not** resume a *new* (batch 8 / compile / different eval cadence) run from this job’s ckpts. After ep1 there *are* `checkpoint_last.pt` / `checkpoint_epoch_01.pt` / `checkpoint_best.pt` (~50.4 MB, optimizer included). Resume is valid **only** to continue *this* CLI (`--batch 4 --accum 8`) after a crash. Shuffle state is not saved; `steps_per_epoch` happens to match 8×4 vs 4×8, but compile/TF32/cudnn algos will not.
- **Do not** re-enable grad checkpoint (disabled on purpose; 5.5 GB free).
- **Do not** jump to batch 16 without a *solo* VRAM probe (Wayland + cudnn workspace on 8 GB).
- **Do not** raise `--num-workers` to 8 on this laptop while util is 95–100%.
- **Do not** `--eval-test` every epoch (already end-only).
- **Do not** treat `sparse_prob` / `mask_ratio` as speed knobs; masked sites still go through dense convs.
- **Do not** kill PID 105207 / any `90_train_convnextv2`.

---

## 5) Live run (PID 105207) — leave it

- Steady-state logs: `epoch N batch {1,2,3, then every 50}` with `gpu_gb=2.34`. Cadence matches ~237 ms/step. Duplicated lines are `log()` **and** `detach_atto.sh` redirecting stdout into the same `run.log`.
- GPU-bound: 95–100% util across 5×1 s samples during ep2; memory controller ~50%. A one-off 78% sample was the ep1→ep2 boundary (eval + worker respawn).
- `h_now fallback zeros` lines are a few clips without spatial json; they repeat each epoch because workers are not persistent. Not a stall.
- Epoch-end cost in this codebase: `collect()` on 5 120 windows (dense `mask=None`, CPU row dicts, no `pin_memory` on sel loader) + `checkpoint_epoch_XX.pt` **and** `checkpoint_last.pt` every epoch, plus `checkpoint_best.pt` when selection rank improves.
- Let it finish 30 epochs. Next run: batch 8, accum 4, eval-every 5 or 10, persistent workers, eval `pin_memory`, TF32.
