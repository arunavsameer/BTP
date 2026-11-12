#!/usr/bin/env python3
"""CPU/GPU smoke: load Trial-3 Pair-CF decoder, one student step. No Trial-3 writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa4.training.optimizer import build_scheduler  # noqa: E402
from collision_jepa4.utils.device import resolve_device  # noqa: E402
from student.losses import latent_loss  # noqa: E402
from student.model import StudentLatPairCF  # noqa: E402
from student.parent import decoder_state_hash, load_parent_decoder  # noqa: E402
from student.paths import EXPECTED_DECODER_HASH, parent_ckpt  # noqa: E402


def main() -> None:
    device = resolve_device("cpu")
    ckpt = parent_ckpt()
    decoder = load_parent_decoder(ckpt)
    got = decoder_state_hash(decoder)
    model = StudentLatPairCF(decoder).to(device)
    audit = model.param_audit()
    clip = torch.rand(2, 16, 3, 256, 256, device=device)
    ft = torch.randn(2, 128, 16, 16, device=device)
    gt = torch.zeros(2, 5, 5, device=device)
    out = model(clip)
    assert out["F"].shape == (2, 128, 16, 16), out["F"].shape
    assert out["Hhat_1s"].shape == (2, 5, 5), out["Hhat_1s"].shape
    mu = ft.mean(dim=(0, 2, 3), keepdim=True)
    sigma = ft.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    batch = {"F_teacher": ft, "gt_1s": gt}
    loss, terms = latent_loss(
        out, batch, feat_w=1.0, cos_w=0.2, mgd_w=0.5, rel_w=0.2, mu=mu, sigma=sigma, predictor=model.predictor
    )
    loss.backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    dec_grad = any(p.grad is not None for p in model.decoder.parameters())
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4
    )
    sch = build_scheduler(opt, 5, 1e-6, warmup_epochs=1)
    opt.step()
    if sch is not None:
        sch.step()
    report = {
        "ok": True,
        "device": device,
        "parent_ckpt": str(ckpt),
        "decoder_hash": got,
        "expected_hash": EXPECTED_DECODER_HASH,
        "hash_ok": got == EXPECTED_DECODER_HASH,
        "param_audit": audit,
        "encoder_got_grad": enc_grad,
        "decoder_got_grad": dec_grad,
        "loss_terms": terms,
        "trial3_untouched": True,
    }
    if dec_grad:
        raise SystemExit("frozen decoder received gradients")
    if not enc_grad:
        raise SystemExit("encoder received no gradients")
    if got != EXPECTED_DECODER_HASH:
        raise SystemExit(f"decoder hash {got} != {EXPECTED_DECODER_HASH}")
    print(json.dumps(report, indent=2))
    print("smoke OK")


if __name__ == "__main__":
    main()
