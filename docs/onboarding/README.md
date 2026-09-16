# Onboarding — start here

You just joined a team building a **collision-warning wearable** for a walking pedestrian (someone who is blind or has low vision). A forehead camera watches the sidewalk. A vest should buzz **left / center / right** a second before you would walk into something.

This folder is the story of what we have actually built so far, in the order a teammate would tell it.

| Read | What it is |
| --- | --- |
| [01 — Dataset generation](01_dataset_generation.md) | Why we simulate streets, how labels are computed, what an episode looks like |
| [02 — Collision-JEPA](02_collision_jepa.md) | First learned model: frozen V-JEPA teacher + tiny student. What we tried, what broke |
| [03 — model_trial_2](03_model_trial_2.md) | Current architecture: LoRA-tuned V-JEPA Giant → distill into a wearable CNN |

You do **not** need computer-vision, robotics, or “JEPA” background. Each note starts from zero and only then adds the next idea.

## The one-paragraph product

The wearable never names “car” or “person”. It predicts a tiny **5×5 danger map one second into the future**, then beeps left / center / right at SAFE / CAUTION / STOP. Training labels come from a **Blender street simulator**, not from humans drawing boxes. The huge video model (V-JEPA 2) is used **only while training**. The thing that would actually sit on the vest is a small CNN.

## How the repo maps onto that story

```
e:\btp\
  blender_sim/          # production dataset generator (what the models train on)
  synth_sim/            # earlier from-scratch renderer (same idea, different engine)
  impl/ + run_collision.py   # older YOLO + box-growth demo (not the learned vest)
  collision_jepa/       # first teacher/student trial
  model_trial_1/        # shuffle experiment (mentioned in 03; not a separate onboarding doc)
  model_trial_2/        # current teacher (Giant + LoRA) → student
```

If a note and the Python disagree, **the Python wins**. These docs are the teammate explanation, not a frozen spec.
