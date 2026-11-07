"""Two-phase teacher trainer: frozen encoder, then optional LoRA."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..evaluation.evaluator import evaluate_loader
from ..losses.composite import TeacherLoss
from ..models.teacher import CollisionTeacher3
from ..reproducibility import software_versions
from ..utils.device import describe_hardware, select_amp_dtype
from ..utils.logging import JsonlLogger
from .checkpoint import extract_teacher_state, load_checkpoint, load_teacher_state, save_checkpoint
from .early_stopping import EarlyStopping
from .engine import (
    assert_finite,
    current_lrs,
    horizon_mse,
    move_batch,
    balanced_warning_better,
    prefork_stop_better,
    recall5_better,
    selection_better,
    teacher_forward,
)
from .bg_regularization import (
    apply_pair_counterfactual_shrink,
    apply_token_mask,
    bg_reg_dropout_active,
    bg_reg_pair_strength,
    calibrate_beta_clean_anchor,
    calibrate_beta_pair_counterfactual,
    calibrate_lambda_pair,
    clean_anchor_drop_strength,
    deterministic_channel_mask,
    pair_cf_ramp_strength,
    pair_consistency_loss,
    shared_pair_dropout_mask,
    token_channel_dropout,
    train_h_weight_mean,
)
from .pair_signals import (
    canonical_pair_indices,
    encode_batch,
    load_mate_h1s,
    load_mate_tokens,
    pair_dir_adv_loss,
    pair_gt_consistent,
    pair_padr_loss,
    pcgrad_grads,
    subset_batch,
    subset_out,
)
from .optimizer import build_param_groups, build_scheduler

logger = logging.getLogger("collision_jepa3.trainer")


class PairBuffer:
    """Stop-grad store of recent Z_t keyed by (background_pair_id, anchor)."""

    def __init__(self, max_items: int = 64):
        self.max_items = int(max_items)
        self._store: OrderedDict[tuple[str, int], tuple[str, torch.Tensor]] = OrderedDict()

    def pop_mate(self, pair_id: str, anchor: int, clip_id: str) -> torch.Tensor | None:
        key = (pair_id, int(anchor))
        if key not in self._store:
            return None
        mate_id, z = self._store[key]
        if mate_id == clip_id:
            return None
        return z

    def put(self, pair_id: str, anchor: int, clip_id: str, z: torch.Tensor) -> None:
        key = (pair_id, int(anchor))
        self._store[key] = (clip_id, z.detach())
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


class TeacherTrainer:
    def __init__(
        self,
        cfg,
        model: CollisionTeacher3,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: str,
        split_fingerprint: str,
        manifest_fingerprint: str,
        preprocess_signature: dict,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.split_fingerprint = split_fingerprint
        self.manifest_fingerprint = manifest_fingerprint
        self.preprocess_signature = preprocess_signature
        self.loss_fn = TeacherLoss(cfg, device=device)
        self.amp_dtype = select_amp_dtype(cfg, device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype is torch.float16) if device.startswith("cuda") else torch.amp.GradScaler("cuda", enabled=False)
        self.pair_buffer = PairBuffer(int(cfg.get("training.max_pair_buffer", 64)))
        self.ckpt_root = cfg.path("paths.checkpoint_root")
        self.log = JsonlLogger(cfg.path("paths.logs_root") / "train.jsonl")
        self.global_step = 0
        self.best_sel: dict[str, Any] | None = None
        self._epoch = 0
        self._padr_lp: float | None = None
        self._padr_lt: float | None = None
        self._grad_diag_epoch = -1
        self._bg_lambda_pair: float | None = None
        self._bg_beta_clean_anchor: float | None = None
        self._bg_beta_pair_cf: float | None = None
        self._pair_cf_U: torch.Tensor | None = None
        self._w_train_mean = 1.0
        self._bg_reg_drop_channels: list[int] = []
        self._init_bg_reg(train_loader.dataset)

    def _init_bg_reg(self, train_ds) -> None:
        from .bg_regularization import load_channel_sets_from_p4

        bg = self.cfg.get("training.bg_reg") or {}
        if not bool(bg.get("enabled")):
            return
        if bool(bg.get("pair_counterfactual_shrink")):
            root = Path(__file__).resolve().parent.parent.parent
            rel = str(bg.get("pair_cf_subspace_npz", "reports/pair_background_counterfactual/subspace/learned_k16.npz"))
            p = Path(rel)
            if not p.is_absolute():
                p = root / p
            U = np.load(p)["U"].astype(np.float32)
            self._pair_cf_U = torch.from_numpy(U)
            logger.info("pair_cf: loaded U shape %s from %s", tuple(U.shape), p)
            return
        if not (
            bool(bg.get("pair_consistency"))
            or bool(bg.get("targeted_dropout"))
            or bool(bg.get("random_dropout"))
            or bool(bg.get("clean_anchored_dropout"))
        ):
            return
        root = Path(__file__).resolve().parent.parent.parent
        p4_rel = str(bg.get("p4_report", "reports/p4_channel_icc/p4_report.json"))
        p4_path = Path(p4_rel)
        if not p4_path.is_absolute():
            p4_path = root / p4_path
        targeted, _n_elig, eligible = load_channel_sets_from_p4(p4_path)
        if bool(bg.get("random_dropout")):
            rng = np.random.default_rng(int(self.cfg.get("project.seed", 1337)))
            k = len(targeted)
            pick = rng.choice(np.array(eligible, dtype=np.int64), size=min(k, len(eligible)), replace=False)
            self._bg_reg_drop_channels = [int(x) for x in pick.tolist()]
        elif bool(bg.get("targeted_dropout")) or bool(bg.get("clean_anchored_dropout")):
            self._bg_reg_drop_channels = list(targeted)
        logger.info(
            "bg_reg: %d dropout channels (targeted=%s random=%s clean_anchor=%s)",
            len(self._bg_reg_drop_channels),
            bg.get("targeted_dropout"),
            bg.get("random_dropout"),
            bg.get("clean_anchored_dropout"),
        )
        if bool(bg.get("pair_consistency")):
            self._w_train_mean = float(train_h_weight_mean(self.cfg, train_ds))
            logger.info("bg_reg w_train_mean=%.6f", self._w_train_mean)
        else:
            self._w_train_mean = 1.0

    def _apply_bg_reg_tokens(self, batch: dict) -> dict:
        bg = self.cfg.get("training.bg_reg") or {}
        if not bool(bg.get("enabled")) or "tokens" not in batch:
            return batch
        if bool(bg.get("clean_anchored_dropout")) or bool(bg.get("pair_counterfactual_shrink")):
            return batch
        if not bg_reg_dropout_active(
            int(self._epoch),
            dropout_on=bool(bg.get("targeted_dropout") or bg.get("random_dropout")),
        ):
            return batch
        ch = self._bg_reg_drop_channels
        if not ch:
            return batch
        tok = batch["tokens"]
        c_dim = int(tok.shape[-1])
        device = tok.device
        dtype = tok.dtype
        p_drop = float(bg.get("dropout_prob", 0.5))
        paired = bool(batch.get("bg_reg_paired", False))
        if paired:
            bsz = int(tok.shape[0])
            mid = bsz // 2
            mask = shared_pair_dropout_mask(
                ch, c_dim=c_dim, device=device, dtype=dtype, p_drop=p_drop
            )
            tok = tok.clone()
            tok[:mid] = apply_token_mask(tok[:mid], mask)
            tok[mid:] = apply_token_mask(tok[mid:], mask)
            batch["tokens"] = tok
        else:
            batch["tokens"] = token_channel_dropout(
                tok, ch, training=True, p_drop=p_drop
            )
        return batch

    def _autocast(self):
        if self.amp_dtype is None or not self.device.startswith("cuda"):
            from contextlib import nullcontext

            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _bg_pair_tensor(self, out, batch) -> torch.Tensor | None:
        z = out["Z_t"]
        mates = []
        valid = batch["bg_pair_valid"]
        valid_list = valid.detach().cpu().tolist() if torch.is_tensor(valid) else list(valid)
        anchors = batch["anchor"]
        anchor_list = anchors.detach().cpu().tolist() if torch.is_tensor(anchors) else list(anchors)
        for i in range(z.shape[0]):
            if not bool(valid_list[i]):
                continue
            pair_id = batch["background_pair_id"][i]
            anchor = int(anchor_list[i])
            clip_id = batch["clip_id"][i]
            mate = self.pair_buffer.pop_mate(pair_id, anchor, clip_id)
            if mate is not None:
                mates.append((i, mate.to(device=z.device, dtype=z.dtype)))
            self.pair_buffer.put(pair_id, anchor, clip_id, z[i])
        if not mates:
            return None
        # Align stored mates to the current batch slots via gathering one pair at a time.
        # Average cosine over available pairs by stacking them as a mini comparison.
        z_cur = torch.stack([z[i] for i, _ in mates], dim=0)
        z_mate = torch.stack([m for _, m in mates], dim=0)
        from ..losses.background_consistency import background_consistency_loss

        return background_consistency_loss(z_cur, z_mate)

    def _dir_adv_weight(self) -> float:
        return float(self.cfg.get("loss.pair_dir_adv_weight", 0.0) or 0.0)

    def _pcgrad_on(self) -> bool:
        return bool(self.cfg.get("training.pair_pcgrad", False))

    def _padr_on(self) -> bool:
        return bool(self.cfg.get("training.padr", False))

    def _padr_scale(self) -> float:
        if not self._padr_on():
            return 0.0
        warm = int(self.cfg.get("training.padr_warmup_epochs", 18))
        ramp = max(1, int(self.cfg.get("training.padr_ramp_epochs", 12)))
        epoch = int(self._epoch)
        if epoch < warm:
            return 0.0
        if epoch < warm + ramp:
            return float(epoch - warm + 1) / float(ramp)
        return 1.0

    def _need_pair_encode(self) -> bool:
        return self._dir_adv_weight() > 0 or (self._padr_on() and self._padr_scale() > 0)

    def _bg_reg_encode_batch(self, batch: dict) -> bool:
        bg = self.cfg.get("training.bg_reg") or {}
        if not bool(bg.get("enabled")) or "tokens" not in batch:
            return False
        if bg_reg_pair_strength(int(self._epoch)) > 0 and bool(bg.get("pair_consistency")):
            return True
        if bg_reg_dropout_active(
            int(self._epoch),
            dropout_on=bool(bg.get("targeted_dropout") or bg.get("random_dropout")),
        ):
            return True
        if bool(bg.get("clean_anchored_dropout")):
            ep = int(self._epoch)
            if clean_anchor_drop_strength(ep) > 0 and bool(batch.get("bg_reg_paired", False)):
                return True
        if bool(bg.get("pair_counterfactual_shrink")):
            if pair_cf_ramp_strength(int(self._epoch)) > 0 and bool(batch.get("bg_reg_paired", False)):
                return False
        return False

    def _pair_cf_pair_pick(self, batch: dict) -> bool:
        bg = self.cfg.get("training.bg_reg") or {}
        if not bool(bg.get("pair_counterfactual_shrink")):
            return False
        if pair_cf_ramp_strength(int(self._epoch)) <= 0:
            return False
        if not bool(batch.get("bg_reg_paired", False)):
            return False
        pair_ids = batch.get("background_pair_id") or []
        if not pair_ids:
            return False
        frac = float(bg.get("pair_cf_pair_frac", 0.5))
        key_pick = f"{int(self._epoch)}|{str(pair_ids[0])}|pair_cf_aux"
        h = hashlib.sha256(key_pick.encode("utf-8")).digest()
        thr = min(255, max(0, int(round(frac * 256))))
        return h[0] < thr

    def _clean_anchor_pair_pick(self, batch: dict) -> bool:
        bg = self.cfg.get("training.bg_reg") or {}
        if not bool(bg.get("clean_anchored_dropout")):
            return False
        if clean_anchor_drop_strength(int(self._epoch)) <= 0:
            return False
        if not bool(batch.get("bg_reg_paired", False)):
            return False
        pair_ids = batch.get("background_pair_id") or []
        if not pair_ids:
            return False
        frac = float(bg.get("clean_anchor_pair_frac", 0.5))
        key_pick = f"{int(self._epoch)}|{str(pair_ids[0])}|clean_anchor_aux"
        h = hashlib.sha256(key_pick.encode("utf-8")).digest()
        thr = min(255, max(0, int(round(frac * 256))))
        return h[0] < thr

    def _clean_anchor_augment_batch(self, batch: dict) -> dict | None:
        bg = self.cfg.get("training.bg_reg") or {}
        ch = self._bg_reg_drop_channels
        if not ch:
            return None
        tok = batch["tokens"]
        bsz = int(tok.shape[0])
        if bsz < 2 or bsz % 2 != 0:
            return None
        mid = bsz // 2
        p_drop = float(bg.get("dropout_prob", 0.25))
        c_dim = int(tok.shape[-1])
        device = tok.device
        dtype = tok.dtype
        anchors = batch["anchor"]
        pair_ids = batch.get("background_pair_id") or [""] * bsz
        ep = int(self._epoch)
        tok_d = tok.clone()
        for i in range(mid):
            anc = int(anchors[i].item()) if torch.is_tensor(anchors[i]) else int(anchors[i])
            mk = f"{ep}|{str(pair_ids[i])}|{anc}"
            mask_c = deterministic_channel_mask(
                ch, c_dim=c_dim, device=device, dtype=dtype, p_drop=p_drop, key=mk
            )
            tok_d[i] = apply_token_mask(tok[i], mask_c)
            tok_d[i + mid] = apply_token_mask(tok[i + mid], mask_c)
        return {**batch, "tokens": tok_d}

    def _forward_loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        batch = move_batch(batch, self.device)
        batch_clean = self._apply_bg_reg_tokens(batch)
        extras: dict[str, float] = {}
        bg = self.cfg.get("training.bg_reg") or {}
        z = fine = None
        out = None
        with self._autocast():
            if self._need_pair_encode() or self._bg_reg_encode_batch(batch_clean):
                z, fine = encode_batch(self.model, batch_clean)
                out = self.model._heads(z, fine)
            else:
                out = teacher_forward(self.model, batch_clean)
            packed = self.loss_fn(out, batch_clean)
            extras.update(self.loss_fn.diag)

            strength = clean_anchor_drop_strength(int(self._epoch))
            extras["clean_anchor_drop_strength"] = float(strength)
            beta = float(self._bg_beta_clean_anchor or 0.0)
            extras["bg_beta_clean_anchor"] = float(beta)
            pick = self._clean_anchor_pair_pick(batch_clean)
            extras["clean_anchor_pair_aux"] = 1.0 if pick else 0.0

            if (
                bool(bg.get("enabled"))
                and bool(bg.get("clean_anchored_dropout"))
                and strength > 0
                and beta > 0
                and pick
            ):
                db = self._clean_anchor_augment_batch(batch_clean)
                if db is not None:
                    out_d = teacher_forward(self.model, db)
                    packed_d = self.loss_fn(out_d, db)
                    l_d = packed_d.total
                    packed.terms["L_drop"] = l_d.detach()
                    packed.terms["L_task"] = packed.total.detach()
                    w_drop = float(beta) * float(strength) * l_d
                    packed.terms["L_drop_weighted"] = w_drop.detach()
                    packed.total = packed.total + w_drop

            cf_strength = pair_cf_ramp_strength(int(self._epoch))
            extras["pair_cf_ramp"] = float(cf_strength)
            beta_cf = float(self._bg_beta_pair_cf or 0.0)
            extras["bg_beta_pair_cf"] = float(beta_cf)
            pick_cf = self._pair_cf_pair_pick(batch_clean)
            extras["pair_cf_aux"] = 1.0 if pick_cf else 0.0
            if (
                bool(bg.get("enabled"))
                and bool(bg.get("pair_counterfactual_shrink"))
                and self._pair_cf_U is not None
                and cf_strength > 0
                and beta_cf > 0
                and pick_cf
            ):
                lam = float(bg.get("pair_cf_lambda", 1.0))
                U = self._pair_cf_U.to(device=batch_clean["tokens"].device, dtype=torch.float32)
                tok_cf = apply_pair_counterfactual_shrink(batch_clean["tokens"], U, lam=lam)
                b_cf = {**batch_clean, "tokens": tok_cf}
                out_cf = teacher_forward(self.model, b_cf)
                l_cf = self.loss_fn(out_cf, b_cf).total
                packed.terms["L_cf"] = l_cf.detach()
                w_cf = float(beta_cf) * float(cf_strength) * l_cf
                packed.terms["L_cf_weighted"] = w_cf.detach()
                packed.total = packed.total + w_cf

            if float(self.loss_fn.bg_w) > 0:
                bg_term = self._bg_pair_tensor(out, batch_clean)
                if bg_term is not None:
                    old_bg = packed.terms.get("L_bg", bg_term.new_zeros(()))
                    packed.terms["L_bg"] = bg_term
                    packed.total = packed.total - self.loss_fn.bg_w * old_bg + self.loss_fn.bg_w * bg_term
            if "L_task" not in packed.terms:
                packed.terms["L_task"] = packed.total.detach()
            if self._dir_adv_weight() > 0 and z is not None and fine is not None:
                extra, stats = self._dir_adv_term(batch_clean, z, fine)
                if extra is not None:
                    packed.terms["L_dir_adv"] = extra
                    packed.total = packed.total + self._dir_adv_weight() * extra
                    extras.update(stats)
            if self._padr_on() and self._padr_scale() > 0 and z is not None and fine is not None:
                lp, lt, stats = self._padr_term(batch_clean, z, fine, out)
                extras.update(stats)
                extras["padr_scale"] = float(self._padr_scale())
                extras["padr_lambda_pair"] = float(self._padr_lp or 0.0)
                extras["padr_lambda_tangent"] = float(self._padr_lt or 0.0)
                if lp is not None and lt is not None:
                    scale = self._padr_scale()
                    lam_p = float(self._padr_lp or 0.0)
                    lam_t = float(self._padr_lt or 0.0)
                    packed.terms["L_pair"] = lp
                    packed.terms["L_tangent"] = lt
                    extra = scale * (lam_p * lp + lam_t * lt)
                    packed.terms["L_padr"] = extra
                    packed.total = packed.total + extra
                    extras["L_padr_weighted"] = float(extra.detach())
            if bool(bg.get("enabled")) and bool(bg.get("pair_consistency")) and out.get("Hhat_1s") is not None:
                pstrength = bg_reg_pair_strength(int(self._epoch))
                extras["bg_reg_pair_strength"] = float(pstrength)
                lam = float(self._bg_lambda_pair or 0.0)
                extras["bg_lambda_pair"] = float(lam)
                if (
                    pstrength > 0
                    and lam > 0
                    and bool(batch_clean.get("bg_reg_paired", False))
                    and "bg_pair_gt_valid" in batch_clean
                ):
                    bsz = int(out["Hhat_1s"].shape[0])
                    mid = bsz // 2
                    ha = out["Hhat_1s"][:mid]
                    hb = out["Hhat_1s"][mid:]
                    pv = batch_clean["bg_pair_gt_valid"][:mid]
                    if bool(pv.any()):
                        l_pair = pair_consistency_loss(
                            ha[pv],
                            hb[pv],
                            batch_clean["h_1s"][:mid][pv],
                            batch_clean["amb_1s"][:mid][pv],
                            w_train_mean=float(self._w_train_mean),
                        )
                        w_pair = float(pstrength) * lam * l_pair
                        packed.terms["L_bg_pair"] = l_pair
                        packed.total = packed.total + w_pair
                        extras["L_bg_pair_weighted"] = float(w_pair.detach())
        if int(self._epoch) != int(self._grad_diag_epoch):
            extras.update(self._loss_grad_diagnostics(packed))
            self._grad_diag_epoch = int(self._epoch)
        assert_finite(packed.total, "L_total")
        floats = packed.as_floats()
        floats.update(extras)
        return packed.total, floats

    def _dir_adv_term(self, batch, z, fine):
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "_load_tokens"):
            return None, {}
        mate_tok, idx_all = load_mate_tokens(dataset, batch, self.device)
        if mate_tok is None or not idx_all:
            return None, {"dir_adv_n": 0.0}
        keep = canonical_pair_indices(batch, idx_all)
        if not keep:
            return None, {"dir_adv_n": 0.0}
        pos = {j: k for k, j in enumerate(idx_all)}
        idx_t = torch.tensor(keep, device=z.device, dtype=torch.long)
        mate_idx = torch.tensor([pos[i] for i in keep], device=z.device, dtype=torch.long)
        _z_m, fine_m = self.model.encode_pair(tokens=mate_tok.index_select(0, mate_idx))
        batch_a = subset_batch(batch, idx_t)
        loss, stats = pair_dir_adv_loss(
            self.model,
            self.loss_fn,
            z_a=z.index_select(0, idx_t),
            fine_a=fine.index_select(0, idx_t),
            fine_b=fine_m,
            batch_a=batch_a,
        )
        return loss, stats

    def _padr_term(self, batch, z, fine, out):
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "_load_tokens"):
            return None, None, {}
        mate_tok, idx_all = load_mate_tokens(dataset, batch, self.device)
        if mate_tok is None or not idx_all:
            return None, None, {"padr_n": 0.0}
        keep = canonical_pair_indices(batch, idx_all)
        if not keep:
            return None, None, {"padr_n": 0.0}
        pos = {j: k for k, j in enumerate(idx_all)}
        idx_t = torch.tensor(keep, device=z.device, dtype=torch.long)
        mate_idx = torch.tensor([pos[i] for i in keep], device=z.device, dtype=torch.long)
        z_m, fine_m = self.model.encode_pair(tokens=mate_tok.index_select(0, mate_idx))
        y = batch["h_1s"].index_select(0, idx_t)
        mask = batch["amb_1s"].index_select(0, idx_t)
        za = out["Hhat_1s_logits"].index_select(0, idx_t)
        gt_tol = float(self.cfg.get("training.padr_gt_tol", 1e-3))
        pair_valid = None
        if bool(self.cfg.get("training.padr_gt_filter", True)):
            gt_b = load_mate_h1s(dataset, batch, idx_t)
            if gt_b is not None:
                pair_valid = pair_gt_consistent(y, gt_b, tol=gt_tol)
        return pair_padr_loss(
            self.model,
            z_a=z.index_select(0, idx_t),
            fine_a=fine.index_select(0, idx_t),
            z_b=z_m,
            fine_b=fine_m,
            y=y,
            mask=mask,
            za=za,
            pair_valid=pair_valid,
            rho_lo=float(self.cfg.get("training.padr_rho_lo", 0.05)),
            rho_hi=float(self.cfg.get("training.padr_rho_hi", 0.10)),
            rms_frac=float(self.cfg.get("training.padr_rms_frac", 0.05)),
        )

    def _grad_norm(self, loss: torch.Tensor, params: list[torch.nn.Parameter], retain: bool = True) -> float:
        if loss is None or not torch.is_tensor(loss) or not loss.requires_grad:
            return 0.0
        grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        total = 0.0
        for g in grads:
            if g is not None:
                total += float(g.detach().float().pow(2).sum())
        return math.sqrt(total)

    def _flat_grads(self, loss: torch.Tensor, params: list[torch.nn.Parameter], retain: bool = True) -> torch.Tensor | None:
        if loss is None or not torch.is_tensor(loss) or not loss.requires_grad:
            return None
        grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        parts = [(g if g is not None else torch.zeros_like(p)).detach().float().reshape(-1) for g, p in zip(grads, params)]
        if not parts:
            return None
        return torch.cat(parts)

    def _loss_grad_diagnostics(self, packed) -> dict[str, float]:
        stats: dict[str, float] = {}
        try:
            adapter = [p for p in self.model.adapter.parameters() if p.requires_grad]
            decoder = [p for p in self.model.decoder.parameters() if p.requires_grad]
            params = adapter + decoder
            if not params:
                return stats
            wmap = {
                "L_1s": 1.0,
                "L_blob": float(self.loss_fn.blob_w),
                "L_stop_hinge": float(self.loss_fn.stop_hinge_w),
                "L_safe_hinge": float(self.loss_fn.safe_hinge_w),
                "L_close": float(self.loss_fn.close_w),
                "L_FP": float(self.loss_fn.fp_w),
                "L_dir": float(self.loss_fn.dir_w),
                "L_pair": float(self._padr_lp or 0.0) * float(self._padr_scale()),
                "L_tangent": float(self._padr_lt or 0.0) * float(self._padr_scale()),
            }
            for name, weight in wmap.items():
                term = packed.terms.get(name)
                if term is None or not torch.is_tensor(term) or float(weight) == 0.0:
                    continue
                weighted = weight * term
                stats[f"g_adapter_{name}"] = self._grad_norm(weighted, adapter, retain=True)
                stats[f"g_decoder_{name}"] = self._grad_norm(weighted, decoder, retain=True)
            flats: dict[str, torch.Tensor] = {}
            pairs = (("L_stop_hinge", "L_1s"), ("L_close", "L_1s"), ("L_blob", "L_stop_hinge"))
            needed = {a for pair in pairs for a in pair}
            for name in needed:
                term = packed.terms.get(name)
                if term is None or not torch.is_tensor(term):
                    continue
                flats[name] = self._flat_grads(term, params, retain=True)
            for a, b in pairs:
                ga, gb = flats.get(a), flats.get(b)
                if ga is None or gb is None:
                    continue
                denom = (ga.norm() * gb.norm()).clamp_min(1e-12)
                stats[f"g_cos_{a}_{b}"] = float((ga @ gb) / denom)
        except Exception as exc:
            logger.warning("grad diagnostics skipped: %s", exc)
        return stats

    def _calibrate_padr(self, phase: str) -> None:
        params = self._trainable_params(phase)
        if not params:
            self._padr_lp = 0.05
            self._padr_lt = 0.05
            return
        g_task: list[float] = []
        g_pair: list[float] = []
        g_tan: list[float] = []
        n_ok = 0
        self.model.train()
        self.model.backbone.eval()
        max_batches = int(self.cfg.get("training.padr_calibrate_batches", 16))
        target_frac = float(self.cfg.get("training.padr_grad_frac", 0.08))
        for batch in self.train_loader:
            batch = move_batch(batch, self.device)
            with self._autocast():
                z, fine = encode_batch(self.model, batch)
                out = self.model._heads(z, fine)
                packed = self.loss_fn(out, batch)
                lp, lt, _stats = self._padr_term(batch, z, fine, out)
            if lp is None or lt is None:
                continue
            g_task.append(self._grad_norm(packed.total, params, retain=True))
            g_pair.append(self._grad_norm(lp, params, retain=True))
            g_tan.append(self._grad_norm(lt, params, retain=False))
            n_ok += 1
            self.model.zero_grad(set_to_none=True)
            if n_ok >= max_batches:
                break
        if not g_task:
            self._padr_lp = 0.05
            self._padr_lt = 0.05
            logger.warning("PADR calibration found no pairs; using λp=λt=0.05")
            return
        med_t = sorted(g_task)[len(g_task) // 2]
        med_p = sorted(g_pair)[len(g_pair) // 2]
        med_n = sorted(g_tan)[len(g_tan) // 2]
        half = 0.5 * target_frac * med_t
        lam_p = half / max(med_p, 1e-8)
        lam_t = half / max(med_n, 1e-8)
        self._padr_lp = float(min(max(lam_p, 0.005), 0.25))
        self._padr_lt = float(min(max(lam_t, 0.005), 0.25))
        logger.info(
            "PADR calibrate n=%d g_task=%.4f g_pair=%.4f g_tan=%.4f λp=%.4f λt=%.4f target_frac=%.2f",
            n_ok,
            med_t,
            med_p,
            med_n,
            self._padr_lp,
            self._padr_lt,
            target_frac,
        )

    def _trainable_params(self, phase: str) -> list[torch.nn.Parameter]:
        return [p for p in self.model.trainable_parameters(include_lora=(phase == "lora")) if p.requires_grad]

    def _pcgrad_backward(self, batch, phase: str) -> dict[str, float]:
        """Original G loss on unpaired samples; PCGrad of (La, Lb) on matched views."""
        batch = move_batch(batch, self.device)
        dataset = getattr(self.train_loader, "dataset", None)
        with self._autocast():
            out = teacher_forward(self.model, batch)
            packed = self.loss_fn(out, batch)
            bg_term = self._bg_pair_tensor(out, batch)
            if bg_term is not None:
                packed.terms["L_bg"] = bg_term
                packed.total = packed.total - self.loss_fn.bg_w * packed.terms["L_bg"].detach() + self.loss_fn.bg_w * bg_term
        terms = packed.as_floats()
        if dataset is None or not hasattr(dataset, "_load_tokens"):
            packed.total.backward()
            return terms
        mate_tok, idx = load_mate_tokens(dataset, batch, self.device)
        if mate_tok is None or not idx:
            packed.total.backward()
            terms["pcgrad_n"] = 0.0
            terms["pcgrad_conflict"] = 0.0
            return terms
        n = out["Hhat_1s"].shape[0]
        idx_t = torch.tensor(idx, device=out["Hhat_1s"].device, dtype=torch.long)
        paired = torch.zeros(n, dtype=torch.bool, device=idx_t.device)
        paired[idx_t] = True
        unpaired = (~paired).nonzero(as_tuple=True)[0]
        batch_a = subset_batch(batch, idx_t)
        out_a = subset_out(out, idx_t)
        with self._autocast():
            la = self.loss_fn(out_a, batch_a).total
            out_b = teacher_forward(self.model, {**batch_a, "tokens": mate_tok})
            lb = self.loss_fn(out_b, batch_a).total
            lu = None
            if int(unpaired.numel()) > 0:
                lu = self.loss_fn(subset_out(out, unpaired), subset_batch(batch, unpaired)).total
        params = self._trainable_params(phase)
        grads, stats = pcgrad_grads(la, lb, lu, params)
        for p, g in zip(params, grads):
            p.grad = g if p.grad is None else p.grad + g
        terms.update(stats)
        terms["pcgrad_n"] = float(len(idx))
        return terms

    def _make_optimizer(self, phase: str, epochs: int):
        include_lora = phase == "lora"
        adapter_params = self.model.adapter_parameters()
        lora_params = self.model.lora_parameters() if include_lora else []
        if include_lora:
            groups = build_param_groups(
                adapter_params,
                lora_params,
                float(self.cfg.get("optimizer.lora_adapter_lr", 0.0002)),
                float(self.cfg.get("optimizer.lora_lr", 0.0001)),
                float(self.cfg.get("optimizer.weight_decay", 0.0001)),
            )
        else:
            groups = build_param_groups(
                adapter_params,
                [],
                float(self.cfg.get("optimizer.adapter_lr", 0.001)),
                float(self.cfg.get("optimizer.lora_lr", 0.0001)),
                float(self.cfg.get("optimizer.weight_decay", 0.0001)),
            )
        opt = torch.optim.AdamW(groups)
        sch = build_scheduler(
            opt,
            epochs,
            float(self.cfg.get("optimizer.min_lr", 1e-6)),
            warmup_epochs=int(self.cfg.get("optimizer.warmup_epochs", 1)),
        )
        return opt, sch

    def _state(self, phase: str, epoch: int, optimizer, scheduler, extra: dict) -> dict[str, Any]:
        blob = extract_teacher_state(self.model)
        blob.update(
            {
                "phase": phase,
                "epoch": epoch,
                "global_step": self.global_step,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": self.scaler.state_dict(),
                "best_selection": self.best_sel,
                "config": self.cfg.to_dict(),
                "backbone_model_id": getattr(self.model.backbone, "model_id", None),
                "backbone_revision": getattr(self.model.backbone, "revision", None),
                "preprocess_signature": self.preprocess_signature,
                "seed": int(self.cfg.get("project.seed", 1337)),
                "split_fingerprint": self.split_fingerprint,
                "manifest_fingerprint": self.manifest_fingerprint,
                "software": software_versions(),
                "parameter_audit": self.model.parameter_audit(),
                "padr_lambda_pair": self._padr_lp,
                "padr_lambda_tangent": self._padr_lt,
                "bg_lambda_pair": self._bg_lambda_pair,
                "bg_beta_clean_anchor": self._bg_beta_clean_anchor,
                "bg_beta_pair_cf": self._bg_beta_pair_cf,
                **extra,
            }
        )
        return blob

    def _disable_token_cache_for_lora(self) -> None:
        """LoRA changes encoder tokens; rebuild pixel loaders if we were on the cache."""
        from ..data.dataset import build_dataloader

        ds_t = getattr(self.train_loader, "dataset", None)
        if ds_t is None or not getattr(ds_t, "use_token_cache", False):
            return
        logger.info("LoRA phase: dropping encoder-token cache and running the backbone")
        ds_t.use_token_cache = False
        self.train_loader = build_dataloader(ds_t, self.cfg, shuffle=True, drop_last=False)
        if self.val_loader is not None:
            ds_v = self.val_loader.dataset
            if hasattr(ds_v, "use_token_cache"):
                ds_v.use_token_cache = False
            self.val_loader = build_dataloader(ds_v, self.cfg, shuffle=False)

    def _validate(self) -> dict[str, Any]:
        if self.val_loader is None:
            return {"selection": {"score": float("nan"), "eligible": False}}
        return evaluate_loader(self.model, self.val_loader, self.cfg, self.device, loss_fn=self.loss_fn)

    def _save(self, name: str, blob: dict) -> Path:
        path = self.ckpt_root / name
        save_checkpoint(blob, path)
        logger.info("wrote %s", path)
        return path

    def run_phase(self, phase: str, epochs: int, resume: dict | None = None) -> dict[str, Any]:
        logger.info("=== phase %s for %d epochs ===", phase, epochs)
        logger.info("parameter audit: %s", self.model.parameter_audit())
        logger.info("hardware: %s", describe_hardware(self.device, getattr(self.model.backbone, "model_id", None)))
        opt, sch = self._make_optimizer(phase, epochs)
        start_epoch = 0
        if resume is not None:
            opt.load_state_dict(resume["optimizer"])
            if sch is not None and resume.get("scheduler"):
                sch.load_state_dict(resume["scheduler"])
            if resume.get("scaler"):
                self.scaler.load_state_dict(resume["scaler"])
            start_epoch = int(resume.get("epoch", 0)) + 1
            self.global_step = int(resume.get("global_step", 0))
            self.best_sel = resume.get("best_selection")
            if resume.get("padr_lambda_pair") is not None:
                self._padr_lp = float(resume["padr_lambda_pair"])
            if resume.get("padr_lambda_tangent") is not None:
                self._padr_lt = float(resume["padr_lambda_tangent"])
            if resume.get("bg_lambda_pair") is not None:
                self._bg_lambda_pair = float(resume["bg_lambda_pair"])
            if resume.get("bg_beta_clean_anchor") is not None:
                self._bg_beta_clean_anchor = float(resume["bg_beta_clean_anchor"])
            if resume.get("bg_beta_pair_cf") is not None:
                self._bg_beta_pair_cf = float(resume["bg_beta_pair_cf"])
            logger.info("resumed %s from epoch %d step %d", phase, start_epoch, self.global_step)
        bg_cfg = self.cfg.get("training.bg_reg") or {}
        if bool(bg_cfg.get("disable_early_stopping")):
            stopper = EarlyStopping(10**9)
        else:
            stopper = EarlyStopping(int(self.cfg.get("training.patience", 4)))
        stop_metric = str(self.cfg.get("training.early_stop_metric", "selection"))
        if stop_metric not in {"selection", "val_mse", "recall_at_5fa", "prefork_stop", "balanced_warning"}:
            raise ValueError(f"unsupported training.early_stop_metric {stop_metric}")
        patience_start = int(self.cfg.get("training.patience_start_epoch", 0))
        post_floor = float(self.cfg.get("training.post_fork_stop_floor", 0.90))
        safe_floor = float(self.cfg.get("training.safe_recall_floor", 0.82))
        stop_floor = float(self.cfg.get("training.stop_recall_floor", 0.75))
        max_safe_stop = float(self.cfg.get("training.max_safe_stop_rate", 0.08))
        prev_val_mse: float | None = None
        accum = max(1, int(self.cfg.get("training.gradient_accumulation_steps", 1)))
        clip_norm = float(self.cfg.get("optimizer.grad_clip_norm", 1.0))
        last_name = f"teacher_phase_{'a' if phase == 'frozen' else 'b'}_last.pt"
        best_name = f"teacher_phase_{'a' if phase == 'frozen' else 'b'}_best.pt"
        history = []
        for epoch in range(start_epoch, epochs):
            self._epoch = epoch
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            bsampler = getattr(self.train_loader, "batch_sampler", None)
            if hasattr(bsampler, "set_epoch"):
                bsampler.set_epoch(epoch)
            self.model.train()
            self.model.backbone.eval()
            if (
                bool((self.cfg.get("training.bg_reg") or {}).get("enabled"))
                and bool((self.cfg.get("training.bg_reg") or {}).get("pair_consistency"))
                and int(epoch) == int((self.cfg.get("training.bg_reg") or {}).get("calibrate_epoch", 3))
                and self._bg_lambda_pair is None
            ):
                logger.info("calibrating bg_reg λ_pair at epoch %d", epoch)
                self._bg_lambda_pair = calibrate_lambda_pair(
                    self.cfg,
                    self.model,
                    self.loss_fn,
                    self.train_loader,
                    self.device,
                    w_train_mean=float(self._w_train_mean),
                    n_batches=int((self.cfg.get("training.bg_reg") or {}).get("calibrate_batches", 50)),
                )
                logger.info("bg_reg λ_pair=%.6f", self._bg_lambda_pair)
            bg_ca = self.cfg.get("training.bg_reg") or {}
            if (
                bool(bg_ca.get("enabled"))
                and bool(bg_ca.get("clean_anchored_dropout"))
                and int(epoch) == int(bg_ca.get("clean_anchor_calibrate_epoch", 1))
                and self._bg_beta_clean_anchor is None
            ):
                logger.info("calibrating clean-anchor β at epoch %d", epoch)
                self._bg_beta_clean_anchor = calibrate_beta_clean_anchor(
                    self.cfg,
                    self.model,
                    self.loss_fn,
                    self.train_loader,
                    self.device,
                    self._bg_reg_drop_channels,
                    calib_epoch=int(bg_ca.get("clean_anchor_calibrate_epoch", 1)),
                    n_batches=int(bg_ca.get("clean_anchor_calibrate_batches", 50)),
                )
                logger.info("clean_anchor β=%.6f", self._bg_beta_clean_anchor)
            if (
                bool(bg_ca.get("enabled"))
                and bool(bg_ca.get("pair_counterfactual_shrink"))
                and self._pair_cf_U is not None
                and int(epoch) == int(bg_ca.get("pair_cf_calibrate_epoch", 1))
                and self._bg_beta_pair_cf is None
            ):
                logger.info("calibrating pair_cf β at epoch %d", epoch)
                self._bg_beta_pair_cf = calibrate_beta_pair_counterfactual(
                    self.cfg,
                    self.model,
                    self.loss_fn,
                    self.train_loader,
                    self.device,
                    self._pair_cf_U,
                    calib_epoch=int(bg_ca.get("pair_cf_calibrate_epoch", 1)),
                    n_batches=int(bg_ca.get("pair_cf_calibrate_batches", 100)),
                )
                logger.info("pair_cf β=%.6f", self._bg_beta_pair_cf)
            if self._padr_on() and self._padr_scale() > 0 and self._padr_lp is None:
                logger.info("calibrating PADR λp, λt at epoch %d", epoch)
                self._calibrate_padr(phase)
            t0 = time.perf_counter()
            running: dict[str, float] = {}
            n_steps = 0
            opt.zero_grad(set_to_none=True)
            max_steps = int(self.cfg.get("training.max_steps", 0) or 0)
            for step, batch in enumerate(self.train_loader):
                if self._pcgrad_on():
                    terms = self._pcgrad_backward(batch, phase)
                    if (step + 1) % accum == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(include_lora=(phase == "lora")), clip_norm)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        self.global_step += 1
                else:
                    loss, terms = self._forward_loss(batch)
                    self.scaler.scale(loss / accum).backward()
                    if (step + 1) % accum == 0:
                        self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(include_lora=(phase == "lora")), clip_norm)
                        self.scaler.step(opt)
                        self.scaler.update()
                        opt.zero_grad(set_to_none=True)
                        self.global_step += 1
                for k, v in terms.items():
                    running[k] = running.get(k, 0.0) + v
                n_steps += 1
                if max_steps > 0 and n_steps >= max_steps:
                    logger.info("stopping epoch early after max_steps=%d", max_steps)
                    break
            if n_steps and (n_steps % accum) != 0:
                if self._pcgrad_on():
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(include_lora=(phase == "lora")), clip_norm)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                else:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(include_lora=(phase == "lora")), clip_norm)
                    self.scaler.step(opt)
                    self.scaler.update()
                    opt.zero_grad(set_to_none=True)
            if sch is not None:
                sch.step()
            train_means = {k: v / max(n_steps, 1) for k, v in running.items()}
            val = self._validate()
            sel = val.get("selection", {})
            mse = horizon_mse(val, self.cfg)
            operating = val.get("operating") or {}
            h1s = (val.get("horizons") or {}).get("1s") or {}
            rec5 = float(operating.get("recall_at_5fa", float("nan")))
            if stop_metric == "val_mse":
                better = prev_val_mse is None or (mse == mse and mse < prev_val_mse)
            elif stop_metric == "recall_at_5fa":
                better = recall5_better(operating, self.best_sel)
            elif stop_metric == "prefork_stop":
                better = prefork_stop_better(h1s, self.best_sel, post_floor=post_floor)
            elif stop_metric == "balanced_warning":
                better = balanced_warning_better(
                    h1s,
                    self.best_sel,
                    post_floor=post_floor,
                    safe_floor=safe_floor,
                    stop_floor=stop_floor,
                    max_safe_stop=max_safe_stop,
                )
            else:
                better = selection_better(sel, self.best_sel)
            blob = self._state(phase, epoch, opt, sch, {"val": val, "train": train_means})
            self._save(last_name, blob)
            if bool(self.cfg.get("training.save_every_epoch", False)):
                self._save(f"teacher_phase_{'a' if phase == 'frozen' else 'b'}_epoch_{epoch:02d}.pt", blob)
            if better:
                tagged = dict(sel)
                tagged["val_mse"] = mse
                tagged["best_epoch"] = epoch
                tagged.update({k: operating.get(k) for k in (
                    "recall_at_5fa", "ap_high", "mean_pred_gt80", "mse",
                    "threshold_5fa", "fa_achieved_5fa", "fa_at_0.5", "recall_at_0.5",
                )})
                tagged.update({k: h1s.get(k) for k in (
                    "pre_fork_stop_recall",
                    "post_fork_stop_recall",
                    "pre_fork_stop_n",
                    "post_fork_stop_n",
                    "post_fork_safe_stop_rate",
                    "safe_recall",
                    "stop_recall",
                    "safe_stop_rate",
                )})
                self.best_sel = tagged
                self._save(best_name, blob)
            dt = time.perf_counter() - t0
            rec = {
                "phase": phase,
                "epoch": epoch,
                "seconds": dt,
                "lrs": current_lrs(opt),
                "train": train_means,
                "val_selection": sel,
                "val_mse": mse,
                "val_operating": operating,
                "val_fork_stop": {
                    "pre_fork_stop_recall": h1s.get("pre_fork_stop_recall"),
                    "post_fork_stop_recall": h1s.get("post_fork_stop_recall"),
                    "post_fork_safe_stop_rate": h1s.get("post_fork_safe_stop_rate"),
                    "safe_recall": h1s.get("safe_recall"),
                    "stop_recall": h1s.get("stop_recall"),
                    "safe_stop_rate": h1s.get("safe_stop_rate"),
                },
                "gpu": describe_hardware(self.device),
            }
            self.log.write(rec)
            history.append(rec)
            logger.info(
                "phase=%s epoch=%d L_total=%.4f val_mse=%.5f rec@5FA=%s AP=%s SAFE=%s STOP=%s FAstop=%s preSTOP=%s postSTOP=%s time=%.1fs",
                phase,
                epoch,
                train_means.get("L_total", float("nan")),
                mse,
                operating.get("recall_at_5fa"),
                operating.get("ap_high"),
                h1s.get("safe_recall"),
                h1s.get("stop_recall"),
                h1s.get("safe_stop_rate"),
                h1s.get("pre_fork_stop_recall"),
                h1s.get("post_fork_stop_recall"),
                dt,
            )
            if stop_metric == "val_mse":
                prev_val_mse = mse
            if epoch < patience_start:
                stopper.step(True)
            elif stopper.step(better):
                keep = None if self.best_sel is None else self.best_sel.get("best_epoch")
                why = {
                    "val_mse": "val_mse",
                    "recall_at_5fa": "recall_at_5fa",
                    "prefork_stop": "pre_fork_stop_recall",
                    "balanced_warning": "min(SAFE,STOP) recall",
                }.get(stop_metric, "val_score")
                logger.info(
                    "early stopping at epoch %d (%s did not improve for %d epochs after warmup); keeping epoch %s",
                    epoch,
                    why,
                    stopper.patience,
                    keep,
                )
                break
        return {"phase": phase, "history": history, "best": self.best_sel}

    def run(self, phase: str = "frozen", resume_path: Path | None = None) -> dict[str, Any]:
        resume = load_checkpoint(resume_path, map_location=self.device) if resume_path else None
        results = {}
        if phase in {"frozen", "both"}:
            results["frozen"] = self.run_phase("frozen", int(self.cfg.get("training.frozen_epochs", 8)), resume if resume and resume.get("phase") == "frozen" else None)
            resume = None
        if phase in {"lora", "both"}:
            self._disable_token_cache_for_lora()
            if not getattr(self.model.backbone, "lora_enabled", False):
                n = self.model.enable_lora()
                logger.info("enabled LoRA on %d linear layers", n)
            lora_resume = resume if resume and resume.get("phase") == "lora" else None
            if lora_resume is None:
                best_a = self.ckpt_root / "teacher_phase_a_best.pt"
                if best_a.exists() and phase in {"lora", "both"}:
                    blob = load_checkpoint(best_a, map_location=self.device)
                    # LoRA just enabled: load adapter/heads from Phase A, ignore missing LoRA.
                    load_teacher_state(
                        self.model,
                        {**blob, "lora": {}},
                        strict=True,
                    )
                    self.best_sel = blob.get("best_selection")
                    logger.info("loaded Phase A best adapter/heads from %s", best_a)
            else:
                load_teacher_state(self.model, lora_resume, strict=True)
            results["lora"] = self.run_phase("lora", int(self.cfg.get("training.lora_epochs", 10)), lora_resume)
        return results
