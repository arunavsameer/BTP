# GPU lock — laptop RTX 4060

**Do not start a second `90_train_convnextv2` or any other CUDA job.**
The training runner owns launches.

Live PID is in `train.pid`. Poll `run.log`; do not exec/foreground another trainer.
