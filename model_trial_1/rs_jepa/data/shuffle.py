"""k×k region shuffle with 180° rotation of opposite cells.

The same permutation is applied to every frame of a clip (and to the future clip)
so temporal differences / looming stay inside a cell. Grid size follows
``student.feature_grid`` (3×3 or 5×5).

Training encodes the *shuffled* mosaic (so the CNN cannot rely on a centre-corridor
prior), then **unpermutes Z** before the predictor/decoder. Prediction therefore
happens in the real layout — objects can move across cells — while the encoder is
still forced to put motion in the cell features themselves.

Eval / deploy never shuffle.

Do not pre-shuffle videos to disk: a clip needs a *new* permutation each time it
is drawn. We shuffle on GPU, once, on the concatenated now+future clip.
"""

from __future__ import annotations

import torch


def rotate_mask_from_perm(perm: torch.Tensor, grid: int) -> torch.Tensor:
    """Bool mask [B, grid*grid] — True where that destination cell should be rot180.

    ``perm[b, i]`` is the index of the source cell that moves to slot ``i``.
    """
    n = grid * grid
    centre = (grid - 1) * 0.5
    device = perm.device
    dst = torch.arange(n, device=device).view(1, n).expand_as(perm)
    src = perm
    src_r = src // grid
    src_c = src % grid
    dst_r = dst // grid
    dst_c = dst % grid
    sr = src_r.float() - centre
    sc = src_c.float() - centre
    dr = dst_r.float() - centre
    dc = dst_c.float() - centre
    moved = src != dst
    opposite = (sr * dr + sc * dc) <= 0
    return moved & opposite


def sample_perm(
    batch_size: int,
    grid: int,
    prob: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a per-item permutation and its 180° mask.

    With probability ``1 - prob`` the permutation is identity (no rotation).
    Vectorized: argsort of uniform keys is a uniform permutation.
    """
    n = grid * grid
    keys = torch.rand(batch_size, n, device=device)
    perm = keys.argsort(dim=1)
    identity = torch.arange(n, device=device).expand(batch_size, n)
    do = torch.rand(batch_size, 1, device=device) < float(prob)
    perm = torch.where(do, perm, identity)
    rotate = rotate_mask_from_perm(perm, grid)
    return perm, rotate


def extract_cells(frames: torch.Tensor, grid: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """frames [B, T, C, H, W] -> cells [B, N, T, C, ch, cw] plus crop box."""
    _b, _t, _c, h, w = frames.shape
    gh = (h // grid) * grid
    gw = (w // grid) * grid
    y0 = (h - gh) // 2
    x0 = (w - gw) // 2
    x = frames[:, :, :, y0 : y0 + gh, x0 : x0 + gw]
    ch, cw = gh // grid, gw // grid
    cells = x.reshape(_b, _t, _c, grid, ch, grid, cw)
    cells = cells.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    cells = cells.reshape(_b, grid * grid, _t, _c, ch, cw)
    return cells, (y0, x0, gh, gw)


def stitch_cells(
    cells: torch.Tensor,
    grid: int,
    out_hw: tuple[int, int],
    crop: tuple[int, int, int, int],
    like: torch.Tensor,
) -> torch.Tensor:
    """cells [B, N, T, C, ch, cw] -> frames [B, T, C, H, W]."""
    b, n, t, c, ch, cw = cells.shape
    y0, x0, gh, gw = crop
    y = cells.reshape(b, grid, grid, t, c, ch, cw)
    y = y.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
    y = y.reshape(b, t, c, gh, gw)
    h, w = out_hw
    if gh == h and gw == w:
        return y
    out = like.clone()
    out[:, :, :, y0 : y0 + gh, x0 : x0 + gw] = y
    return out


def apply_shuffle(
    frames: torch.Tensor,
    perm: torch.Tensor,
    rotate_mask: torch.Tensor,
    grid: int,
) -> torch.Tensor:
    """Permute k×k cells of ``frames`` [B, T, C, H, W]; rot180 where masked.

    ``perm[b, i]`` is the source cell index placed into destination slot ``i``.
    """
    cells, crop = extract_cells(frames, grid)
    b, n, t, c, ch, cw = cells.shape
    idx = perm.view(b, n, 1, 1, 1, 1).expand(-1, -1, t, c, ch, cw)
    shuf = torch.gather(cells, 1, idx)
    rot = torch.rot90(shuf, 2, dims=(-2, -1))
    mask = rotate_mask.view(b, n, 1, 1, 1, 1)
    shuf = torch.where(mask, rot, shuf)
    return stitch_cells(shuf, grid, frames.shape[-2:], crop, frames)


def apply_shuffle_pair(
    frames: torch.Tensor,
    frames_future: torch.Tensor,
    perm: torch.Tensor,
    rotate_mask: torch.Tensor,
    grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One extract/gather/stitch for now+future (same permutation)."""
    n = frames.shape[1]
    both = torch.cat([frames, frames_future], dim=1)
    shuf = apply_shuffle(both, perm, rotate_mask, grid)
    return shuf[:, :n], shuf[:, n:]


def permute_grid(hm: torch.Tensor, perm: torch.Tensor, grid: int) -> torch.Tensor:
    """Apply the same cell permutation to a [B, grid, grid] heatmap."""
    flat = hm.reshape(hm.shape[0], grid * grid)
    out = torch.gather(flat, 1, perm)
    return out.reshape(hm.shape[0], grid, grid)


def unpermute_grid(hm: torch.Tensor, perm: torch.Tensor, grid: int) -> torch.Tensor:
    inv = torch.argsort(perm, dim=1)
    return permute_grid(hm, inv, grid)


def permute_map(z: torch.Tensor, perm: torch.Tensor, grid: int) -> torch.Tensor:
    """Permute cells of a feature map [B, C, grid, grid]."""
    b, c, g, w = z.shape
    if g != grid or w != grid:
        raise ValueError(f"expected square {grid}, got {g}x{w}")
    flat = z.reshape(b, c, grid * grid)
    idx = perm.unsqueeze(1).expand(-1, c, -1)
    return torch.gather(flat, 2, idx).reshape(b, c, grid, grid)


def unpermute_map(z: torch.Tensor, perm: torch.Tensor, grid: int) -> torch.Tensor:
    inv = torch.argsort(perm, dim=1)
    return permute_map(z, inv, grid)
