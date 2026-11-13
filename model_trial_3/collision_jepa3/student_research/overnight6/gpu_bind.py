"""Select a working GPU *before* importing torch in the parent process."""

from __future__ import annotations

import os
import subprocess
import sys


def _ranked_indices() -> list[int]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        cands = []
        for line in raw.strip().splitlines():
            idx, mem = [x.strip() for x in line.split(",")]
            cands.append((float(mem), int(idx)))
        cands.sort()
        return [i for _, i in cands]
    except Exception:
        return list(range(8))


def _probe(idx: int) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(idx)
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize(); print('ok')",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    return r.returncode == 0 and "ok" in (r.stdout or "")


def bind_cuda(pref: str = "auto") -> str:
    """Set CUDA_VISIBLE_DEVICES to one allocatable GPU. Returns physical index as str."""
    already = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if already and already != "auto":
        return already.split(",")[0]
    wanted = None
    p = str(pref or "auto")
    if p.startswith("cuda:"):
        wanted = int(p.split(":")[1])
    elif p.isdigit():
        wanted = int(p)
    order = _ranked_indices()
    if wanted is not None:
        order = [wanted] + [i for i in order if i != wanted]
    tried = []
    for i in order:
        tried.append(i)
        if _probe(i):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            print(f"bound CUDA_VISIBLE_DEVICES={i}", flush=True)
            return str(i)
    raise SystemExit(f"no allocatable CUDA device (probed {tried})")
