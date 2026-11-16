# SAFE / STOP evaluation formula

This scores a **student 5×5 heatmap** against the **ground-truth (GT) occupancy map 1 second later**. It is not the old peak-only class check. Class still comes from the GT peak so we know which frames are STOP, SAFE, or THREAT. The grade itself looks at where the heat sits, how strong it is, and whether the last few seconds rise or fall the same way as GT.

A **frame** is one 5×5 map (25 cells). `peak` = the largest of those 25 cells.

---

## 1. What STOP, SAFE, and THREAT frames are

The **GT class of a frame** uses only the **GT peak** (not the student peak):

- **SAFE** if `peak < 0.25`
- **THREAT** if `0.25 ≤ peak < 0.60`
- **STOP** if `peak ≥ 0.60`

```
gt_class = SAFE    if gt_peak < 0.25
gt_class = THREAT  if 0.25 ≤ gt_peak < 0.60
gt_class = STOP    if gt_peak ≥ 0.60
```

In words: a cold GT map is SAFE; a very hot cell somewhere is STOP; the middle band is THREAT.

The student’s own peak is shown as `pred_class` on the HUD. Clip SAFE and clip STOP do **not** use the student class.

---

## 2. What happens on STOP frames

**What we want:** strong student heat on the real danger blob, not on the rest of the grid, and the last ~3 seconds rising or falling with GT.

### Allowed vs blue

A GT cell is **red** (core blob) if it is ≥ **0.50**.

**Allowed** = red cells plus a one-cell pad. Distance is **Chebyshev 1**: the 3×3 square around each red cell, **including diagonals**. Student heat on allowed cells is not a location error.

**Blue** = every cell outside that pad. Student heat ≥ 0.50 on blue is a **false hot** (leak).

```
red     = (GT cell ≥ 0.50)
allowed = red, then grow by Chebyshev radius 1 (3×3 square)
blue    = not allowed
```

Example: only the center GT cell is ≥ 0.50. `R` = red, `A` = allowed pad, `B` = blue.

```
B B B B B
B A A A B
B A R A B
B A A A B
B B B B B
```

A hot student cell on a corner `A` still covers the blob. A hot student cell on a `B` is leak.

### Coverage

Student max on **allowed** cells, scaled by the STOP cut 0.60. A neighborhood peak of 0.60 or more is full credit.

```
coverage = clip(max(pred on allowed) / 0.60, 0, 1)
```

`clip(..., 0, 1)` means: below 0 becomes 0, above 1 becomes 1.

### Leak

Fraction of **blue** cells where the student is ≥ **0.50** (same cut as a core red blob).

```
leak = (number of blue cells with pred ≥ 0.50) / (number of blue cells)
```

If there are no blue cells, `leak = 0`.

### Spatial

```
spatial = clip(coverage * (1 - leak), 0, 1)
```

If **allowed is empty** (no GT cell ≥ 0.50):

```
spatial = 0
coverage = 0
leak = 1
```

On a true STOP frame this should not happen, because STOP means some GT cell is ≥ 0.60, which is also ≥ 0.50.

### Intent

Six samples of “peak in the allowed neighborhood” (or the full 5×5 if that neighborhood is empty):

```
offsets = (-90, -72, -54, -36, -18, 0)   # frames; about 3 s of history, including now
```

At each offset `tau`, compare student at `tau` with GT at `tau + 1 s` (30 frames). The series value is the max **inside the allowed mask of that GT map**.

```
slope = last sample − first sample
sign  = +1 if slope > 0.05    (rising)
        −1 if slope < −0.05   (falling)
         0 if |slope| ≤ 0.05  (flat; 0.02 jitter is ignored)
```

Then GT sign vs student sign:

```
intent = 1.0  if same sign, or both flat
intent = 0.0  if opposite (one rising, one falling)
intent = 0.5  if one is flat and the other is moving
```

If there are fewer than two usable samples, `intent = 0.5`.

### Per-frame STOP quality

Spatial is 70%; intent is 30%.

```
quality = clip(0.70 * spatial + 0.30 * intent, 0, 1)
```

---

## 3. What happens on SAFE frames

**What we want:** the student map stays **cold**. A high peak anywhere is bad. Spreading medium heat over many cells is also bad. Intent should stay flat, or at least not disagree on rising vs falling.

SAFE spatial does **not** use allowed/blue. The whole 5×5 should be quiet. Intent uses the **full-grid peak** (no blob neighborhood), because there is no red blob.

### Peak term

Full credit if the student’s max is still in the SAFE band (≤ 0.25). Linearly lose credit until the peak hits the STOP cut (0.60). That span is 0.60 − 0.25 = **0.35**.

```
peak_term = clip(1 - relu(max(pred) - 0.25) / 0.35, 0, 1)
```

`relu(x)` is `max(x, 0)`: no extra reward for being colder than 0.25; penalty only starts above 0.25.

- `max(pred) ≤ 0.25` → `peak_term = 1`
- `max(pred) = 0.60` → `peak_term = 0`
- in between, a straight line down

### Hot-cell fraction

A cell is too warm for SAFE at **0.25**. This is the fraction of all 25 cells at or above that cut.

```
frac = (number of cells with pred ≥ 0.25) / 25
```

### Spatial

```
spatial = clip(peak_term * (1 - frac), 0, 1)
```

Cold everywhere → `peak_term = 1`, `frac = 0`, `spatial = 1`. One cell at 0.40 cuts both the peak term and the fraction.

### Intent

Same 6-sample recipe and deadzone **0.05**, but each sample is `max` over the **whole** 5×5. Then the same intent table as STOP (1 / 0 / 0.5). Fewer than two samples → `intent = 0.5`.

### Per-frame SAFE quality

```
quality = clip(0.70 * spatial + 0.30 * intent, 0, 1)
```

---

## 4. What about THREAT frames

THREAT is the **middle band**: not clearly SAFE, not clearly STOP (`0.25 ≤ GT peak < 0.60`).

These frames are **labeled on the HUD** so you can watch the in-between cases. They are **not** averaged into clip SAFE or clip STOP.

Why: mixing them in would blur two different jobs (stay cold vs cover the blob). In this band a 0.50 “red” cut may also leave **no** allowed cells, so STOP-style coverage is a poor fit. Keep the two clip scores pure.

The scorer still *computes* a spatial number on THREAT with the STOP-style formula for the HUD row. That number is **not** folded into the video’s SAFE or STOP means.

---

## 5. How those frame scores become two clip numbers

Score every student time `t` that has 3 s of history (`t ≥ 90`) and a GT map at `t + 1 s`. Each of those times is one **frame quality**, tagged by **GT class**.

Then, for the whole clip / video:

```
clip STOP = mean of quality on GT-STOP frames
            (N/A if there are zero STOP frames)

clip SAFE = mean of quality on GT-SAFE frames
            (N/A if there are zero SAFE frames)
```

THREAT frames are counted for the HUD and for “how many frames,” but they do **not** enter either mean.

A video therefore reports **two** numbers (each 0–1, or N/A): how well the student handled **real STOP** moments, and how well it stayed **SAFE** when GT was cold.
