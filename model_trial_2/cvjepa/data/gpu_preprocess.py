"""GPU-side normalize + clip augment for the student."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def frames_uint8_to_nchw(frames: torch.Tensor) -> torch.Tensor:
    x = frames.permute(0, 1, 4, 2, 3).contiguous()
    return x.to(dtype=torch.float32).mul_(1.0 / 255.0)


def gpu_clip_augment(
    frames: torch.Tensor,
    heatmaps: list[torch.Tensor],
    brightness: float = 0.25,
    contrast: float = 0.25,
    blur_prob: float = 0.2,
    noise_std: float = 6.0,
    flip_prob: float = 0.5,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    bsz, t, c, h, w = frames.shape
    device = frames.device
    b = 1.0 + (torch.rand(bsz, 1, 1, 1, 1, device=device) * 2.0 - 1.0) * brightness
    k = 1.0 + (torch.rand(bsz, 1, 1, 1, 1, device=device) * 2.0 - 1.0) * contrast
    mean = frames.mean(dim=(1, 2, 3, 4), keepdim=True)
    frames = (frames - mean) * k + mean * b
    if noise_std > 0:
        frames = frames + torch.randn_like(frames) * (noise_std / 255.0)
    frames = frames.clamp_(0.0, 1.0)
    if blur_prob > 0:
        blur_m = torch.rand(bsz, device=device) < blur_prob
        if bool(blur_m.any()):
            y = frames.reshape(bsz * t, c, h, w)
            y = F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
            y = y.reshape(bsz, t, c, h, w)
            frames = torch.where(blur_m.view(bsz, 1, 1, 1, 1), y, frames)
    if flip_prob > 0:
        flip_m = torch.rand(bsz, device=device) < flip_prob
        if bool(flip_m.any()):
            frames = torch.where(flip_m.view(bsz, 1, 1, 1, 1), frames.flip(-1), frames)
            flipped = []
            for hm in heatmaps:
                view = flip_m.view((bsz,) + (1,) * (hm.dim() - 1))
                flipped.append(torch.where(view, hm.flip(-1), hm))
            heatmaps = flipped
    return frames, heatmaps


def prepare_batch(batch: dict, device: str | torch.device, *, augment: bool) -> dict:
    out = {}
    for key, val in batch.items():
        out[key] = val.to(device, non_blocking=True) if torch.is_tensor(val) else val
    frames = frames_uint8_to_nchw(out["frames"])
    h_now, h_fut, h_mid = out["h_now"].float(), out["h_future"].float(), out["h_mid"].float()
    extras = []
    extra_keys = []
    for key in ("z_t", "z_plus"):
        if key in out and torch.is_tensor(out[key]):
            extras.append(out[key].float())
            extra_keys.append(key)
    if augment:
        frames, hms = gpu_clip_augment(frames, [h_now, h_fut, h_mid, *extras])
        h_now, h_fut, h_mid = hms[0], hms[1], hms[2]
        for i, key in enumerate(extra_keys):
            out[key] = hms[3 + i]
    out["frames"] = frames
    out["h_now"] = h_now
    out["h_future"] = h_fut
    out["h_mid"] = h_mid
    return out
