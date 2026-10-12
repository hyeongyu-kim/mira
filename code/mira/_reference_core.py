"""Shared BeTTA core utilities for *_laplace_betta methods.

Keep all method-independent code here:
- last-layer Gaussian MC wrapper
- sigma-grid selection
- posterior predictive HE/EH/MI computation
- BeTTA trace logging
- common norm-only adaptation configuration
"""

from contextlib import contextmanager
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. Small utilities
# =============================================================================


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Categorical entropy for probability tensors. Returns shape p.shape[:-1]."""
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def find_last_linear(model: nn.Module) -> nn.Linear:
    """Return the last nn.Linear encountered in model.modules()."""
    last = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        raise RuntimeError("BeTTA requires at least one nn.Linear classifier.")
    return last


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower().replace("-", "_")


def env_float_list(name: str, default):
    """Parse comma/space-separated float grid from env."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return [float(x) for x in default]
    raw = raw.replace(",", " ")
    return [float(x) for x in raw.split() if x.strip()]


def env_int_list(name: str, default):
    """Parse comma/space-separated int grid from env."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return [int(x) for x in default]
    raw = raw.replace(",", " ")
    return [int(float(x)) for x in raw.split() if x.strip()]


# =============================================================================
# 2. Last-layer Gaussian perturbation wrapper
# =============================================================================


class LastLayerGaussianSampler(nn.Module):
    """Wrap a classifier and expose last-layer MC sampling utilities.

    This class is method-agnostic. It does not know about Tent/DeYO/ROID.

    Effective weight noise is always:
        W_sample = W + eps * sigma * scale(W)

    Supported scale modes:
      absolute/fixed/unit/scalar     -> scale = 1
      weight_abs/relative/abs/weight -> scale = |W_ij|
      layer_rms/weight_rms/rms       -> scale = RMS(W)
      layer_mean_abs/mean_abs        -> scale = mean(|W|)
      fan_in/lecun/din               -> scale = sqrt(1 / D_in)
      fan_out/dout                   -> scale = sqrt(1 / D_out)
      inv_fan_in/1_over_din          -> scale = 1 / D_in
      inv_fan_out/1_over_dout        -> scale = 1 / D_out
      he_in/kaiming_in               -> scale = sqrt(2 / D_in)
      he_out/kaiming_out             -> scale = sqrt(2 / D_out)
      xavier/glorot                  -> scale = sqrt(2 / (D_in + D_out))
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        relative_floor: float = 1e-12,
        sigma_scale_mode: str = "absolute",
        bias_scale_mode: str = "auto",
        num_classes: int | None = None,
    ):
        super().__init__()
        self.base = base_model
        self.last_layer = find_last_linear(base_model)
        self.relative_floor = float(relative_floor)
        self.sigma_scale_mode = str(sigma_scale_mode).lower().replace("-", "_")
        self.bias_scale_mode = str(bias_scale_mode).lower().replace("-", "_")

        # Optional anisotropic posterior scale.  The selected sigma remains a scalar,
        # but the effective per-weight std can be factorized as:
        #   std_cd = sigma * scale(W_cd) * aniso_factor_cd.
        # Modes:
        #   scalar/none/off       -> current behavior
        #   class/row            -> one factor per output class
        #   feature/col          -> one factor per feature dimension, from current batch features
        #   factorized/class_feature -> class factor x feature factor
        #   elementwise/weight   -> one deterministic factor per W_cd from |W_cd|
        self.posterior_scale_mode = os.environ.get("BETTA_POSTERIOR_SCALE_MODE", "scalar").strip().lower().replace("-", "_")
        self.aniso_strength = float(os.environ.get("BETTA_ANISO_STRENGTH", "1.0"))
        self.aniso_rank = int(float(os.environ.get("BETTA_ANISO_RANK", "1")))
        self.aniso_min = float(os.environ.get("BETTA_ANISO_MIN", "0.25"))
        self.aniso_max = float(os.environ.get("BETTA_ANISO_MAX", "4.0"))

        self.std_w = None
        self.std_b = None
        self.last_selected_sigma = None
        self.last_aniso_w_factor = None
        self.last_aniso_b_factor = None
        self.last_aniso_class_factor = None
        self.last_aniso_feature_factor = None

        out_dim = int(self.last_layer.weight.shape[0])
        if num_classes is not None and int(num_classes) != out_dim:
            raise ValueError(f"num_classes={num_classes}, but last linear has {out_dim} outputs.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    @contextmanager
    def _capture_last_features(self):
        box = {}

        def hook(_module, inputs):
            box["features"] = inputs[0]

        handle = self.last_layer.register_forward_pre_hook(hook, with_kwargs=False)
        try:
            yield box
        finally:
            handle.remove()

    def extract_last_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through base model once and return the last-linear input."""
        with self._capture_last_features() as box:
            _ = self.base(x)
        feats = box.get("features", None)
        if feats is None:
            raise RuntimeError("Failed to capture last-layer features.")
        return feats.flatten(1) if feats.ndim > 2 else feats

    def _scalar_like_weight(self, value: float, W: torch.Tensor) -> torch.Tensor:
        return W.new_tensor(float(value))

    def _weight_scale(self, W: torch.Tensor, mode: str) -> torch.Tensor:
        mode = str(mode).lower().replace("-", "_")
        C, D = W.shape  # D_out, D_in

        if mode in {"absolute", "fixed", "unit", "scalar", "none", "default"}:
            return torch.ones_like(W)
        if mode in {"weight_abs", "relative", "abs", "weight"}:
            return W.detach().abs().clamp_min(self.relative_floor)
        if mode in {"layer_rms", "weight_rms", "rms"}:
            return self._scalar_like_weight(W.detach().pow(2).mean().sqrt().clamp_min(self.relative_floor), W)
        if mode in {"layer_mean_abs", "weight_mean_abs", "mean_abs"}:
            return self._scalar_like_weight(W.detach().abs().mean().clamp_min(self.relative_floor), W)
        if mode in {"fan_in", "lecun", "din", "1_over_sqrt_din"}:
            return self._scalar_like_weight(math.sqrt(1.0 / max(D, 1)), W)
        if mode in {"fan_out", "dout", "1_over_sqrt_dout"}:
            return self._scalar_like_weight(math.sqrt(1.0 / max(C, 1)), W)
        if mode in {"inv_fan_in", "1_over_din"}:
            return self._scalar_like_weight(1.0 / max(D, 1), W)
        if mode in {"inv_fan_out", "1_over_dout"}:
            return self._scalar_like_weight(1.0 / max(C, 1), W)
        if mode in {"he_in", "kaiming_in"}:
            return self._scalar_like_weight(math.sqrt(2.0 / max(D, 1)), W)
        if mode in {"he_out", "kaiming_out"}:
            return self._scalar_like_weight(math.sqrt(2.0 / max(C, 1)), W)
        if mode in {"xavier", "glorot"}:
            return self._scalar_like_weight(math.sqrt(2.0 / max(C + D, 1)), W)

        raise ValueError(f"Unknown sigma_scale_mode={mode}")

    def _bias_scale(self, b: torch.Tensor | None, w_scale: torch.Tensor) -> torch.Tensor | None:
        if b is None:
            return None

        mode = self.bias_scale_mode
        b_detached = b.detach()

        if mode == "auto":
            if self.sigma_scale_mode in {"weight_abs", "relative", "abs", "weight"}:
                return b_detached.abs().clamp_min(self.relative_floor)
            return w_scale.mean() if torch.is_tensor(w_scale) and w_scale.ndim > 0 else w_scale
        if mode in {"none", "zero"}:
            return torch.zeros_like(b_detached)
        if mode in {"bias_abs", "relative", "abs"}:
            return b_detached.abs().clamp_min(self.relative_floor)
        if mode in {"same", "weight", "weight_mean"}:
            return w_scale.mean() if torch.is_tensor(w_scale) and w_scale.ndim > 0 else w_scale
        if mode in {"absolute", "fixed", "unit", "scalar", "none", "default"}:
            return torch.ones_like(b_detached)

        # Allow bias_scale_mode to reuse any weight scale mode.
        s = self._weight_scale(self.last_layer.weight.detach(), mode)
        return s.mean() if torch.is_tensor(s) and s.ndim > 0 else s

    def _normalize_aniso_factor(self, factor: torch.Tensor) -> torch.Tensor:
        """Normalize an anisotropic factor to mean 1 and clip for stability."""
        factor = torch.where(torch.isfinite(factor), factor, torch.ones_like(factor))
        factor = factor.clamp_min(self.relative_floor)
        factor = factor / factor.mean().clamp_min(self.relative_floor)
        strength = float(getattr(self, "aniso_strength", 1.0))
        if strength != 1.0:
            # strength=0 recovers scalar; strength=1 uses the full normalized factor.
            factor = 1.0 + strength * (factor - 1.0)
        lo = float(getattr(self, "aniso_min", 0.25))
        hi = float(getattr(self, "aniso_max", 4.0))
        return factor.clamp(min=lo, max=hi)

    def _rank_factorized_aniso_factor(
        self,
        W_abs: torch.Tensor,
        class_factor: torch.Tensor,
        feature_factor: torch.Tensor,
        b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cheap low-rank factorized proxy for anisotropic last-layer posterior std.

        rank=1 is the usual class x feature outer product. Higher ranks add a few
        deterministic rank-1 bases derived only from last-layer weights/biases and
        current batch feature activation magnitudes. This avoids an expensive SVD of
        a large ImageNet classifier on every test batch.
        """
        rank = int(max(1, getattr(self, "aniso_rank", 1)))
        rank = min(rank, 4)

        weight_feature = self._normalize_aniso_factor(W_abs.pow(2).mean(dim=0).sqrt())
        if b is not None:
            bias_class = self._normalize_aniso_factor(b.detach().abs().clamp_min(self.relative_floor))
        else:
            bias_class = self._normalize_aniso_factor(W_abs.mean(dim=1))

        components = [
            class_factor[:, None] * feature_factor[None, :],
            class_factor[:, None] * weight_feature[None, :],
            bias_class[:, None] * feature_factor[None, :],
            bias_class[:, None] * weight_feature[None, :],
        ]
        raw = torch.stack(components[:rank], dim=0).mean(dim=0)
        return self._normalize_aniso_factor(raw)

    def _anisotropic_factors(self, feats: torch.Tensor | None, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return (W_factor, class_factor, feature_factor) for anisotropic posterior std.

        W_factor is broadcast-compatible with W [C, D].  class_factor is [C]
        when available; feature_factor is [D] when available.
        """
        mode = str(getattr(self, "posterior_scale_mode", "scalar")).lower().replace("-", "_")
        if mode in {"", "none", "off", "scalar", "global"}:
            return torch.ones_like(W), None, None

        W_abs = W.detach().abs().clamp_min(self.relative_floor)
        C, D = W_abs.shape
        b = self.last_layer.bias

        class_factor = None
        feature_factor = None

        factorized_modes = {
            "factorized", "class_feature", "feature_class", "factorized_grid", "multi_grid", "rank_factorized"
        }

        if mode in {"class", "row", "classwise", "class_wise"} | factorized_modes:
            # Row-wise classifier norm: classes with larger classifier uncertainty/sensitivity
            # receive larger perturbation scale.  Normalized to avoid changing the global sigma.
            class_factor = self._normalize_aniso_factor(W_abs.pow(2).mean(dim=1).sqrt())  # [C]

        if mode in {"feature", "col", "featurewise", "feature_wise"} | factorized_modes:
            if feats is None:
                feature_factor = self._normalize_aniso_factor(W_abs.pow(2).mean(dim=0).sqrt())  # fallback [D]
            else:
                feats_det = feats.detach().to(device=W.device, dtype=W.dtype)
                if feats_det.ndim > 2:
                    feats_det = feats_det.flatten(1)
                # Batch activation magnitude proxy. Dimensions that are more active in
                # the shifted test batch receive larger posterior perturbation.
                feature_factor = self._normalize_aniso_factor(feats_det.abs().mean(dim=0).clamp_min(self.relative_floor))  # [D]

        if mode in {"elementwise", "weight", "per_weight", "full", "diag", "diagonal"}:
            w_factor = self._normalize_aniso_factor(W_abs)
            class_factor = self._normalize_aniso_factor(w_factor.mean(dim=1))
            feature_factor = self._normalize_aniso_factor(w_factor.mean(dim=0))
            return w_factor, class_factor, feature_factor

        if mode in {"class", "row", "classwise", "class_wise"}:
            return class_factor[:, None].expand_as(W), class_factor, None

        if mode in {"feature", "col", "featurewise", "feature_wise"}:
            return feature_factor[None, :].expand_as(W), None, feature_factor

        if mode in factorized_modes:
            w_factor = self._rank_factorized_aniso_factor(W_abs, class_factor, feature_factor, b=b)
            return w_factor, class_factor, feature_factor

        raise ValueError(
            f"Unknown BETTA_POSTERIOR_SCALE_MODE={mode}. Supported: scalar, class, feature, factorized, elementwise."
        )

    def _noise_scales(self, feats: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        W = self.last_layer.weight.detach()
        b = self.last_layer.bias
        w_scale = self._weight_scale(W, self.sigma_scale_mode)
        b_scale = self._bias_scale(b, w_scale)

        aniso_w, class_factor, feature_factor = self._anisotropic_factors(feats, W)
        mode = str(getattr(self, "posterior_scale_mode", "scalar")).lower().replace("-", "_")
        if mode not in {"", "none", "off", "scalar", "global"}:
            w_scale = w_scale * aniso_w
            if b_scale is not None and class_factor is not None:
                b_scale = b_scale * class_factor

        self.last_aniso_w_factor = aniso_w.detach()
        self.last_aniso_b_factor = None if class_factor is None else class_factor.detach()
        self.last_aniso_class_factor = None if class_factor is None else class_factor.detach()
        self.last_aniso_feature_factor = None if feature_factor is None else feature_factor.detach()
        return w_scale, b_scale

    def mc_logits_from_features(self, feats: torch.Tensor, sigma, K: int) -> torch.Tensor:
        """Return MC logits with shape [K, B, C] for one sigma."""
        W, b = self.last_layer.weight, self.last_layer.bias
        feats = feats.to(device=W.device, dtype=W.dtype)
        sigma = torch.as_tensor(sigma, device=W.device, dtype=W.dtype)
        w_scale, b_scale = self._noise_scales(feats)

        base_logits = F.linear(feats, W, b)  # [B, C]

        eps_w = torch.randn((int(K), *W.shape), device=W.device, dtype=W.dtype)
        noise_logits = torch.einsum("bd,kcd->kbc", feats, eps_w * w_scale)
        logits = base_logits.unsqueeze(0) + sigma * noise_logits

        if b is not None:
            eps_b = torch.randn((int(K), b.numel()), device=W.device, dtype=W.dtype)
            logits = logits + sigma * (eps_b * b_scale).unsqueeze(1)

        self.std_w = sigma.detach() * w_scale
        self.std_b = None if b_scale is None else sigma.detach() * b_scale
        self.last_selected_sigma = float(sigma.detach().item()) if sigma.ndim == 0 else None
        return logits

    def mc_logits_grid_from_features(
        self,
        feats: torch.Tensor,
        sigmas: torch.Tensor,
        K: int,
        same_noise: bool = True,
    ) -> torch.Tensor:
        """Return MC logits with shape [S, K, B, C] for a sigma grid."""
        W, b = self.last_layer.weight, self.last_layer.bias
        feats = feats.to(device=W.device, dtype=W.dtype)
        sigmas = torch.as_tensor(sigmas, device=W.device, dtype=W.dtype).flatten()
        w_scale, b_scale = self._noise_scales(feats)
        base_logits = F.linear(feats, W, b)  # [B, C]

        if same_noise:
            eps_w = torch.randn((int(K), *W.shape), device=W.device, dtype=W.dtype)
            noise_logits = torch.einsum("bd,kcd->kbc", feats, eps_w * w_scale)  # [K, B, C]
            logits = base_logits[None, None] + sigmas[:, None, None, None] * noise_logits[None]

            if b is not None:
                eps_b = torch.randn((int(K), b.numel()), device=W.device, dtype=W.dtype)
                logits = logits + sigmas[:, None, None, None] * (eps_b * b_scale)[None, :, None, :]
            return logits

        S = sigmas.numel()
        eps_w = torch.randn((S, int(K), *W.shape), device=W.device, dtype=W.dtype)
        noise_logits = torch.einsum("bd,skcd->skbc", feats, eps_w * w_scale[None, None])
        logits = base_logits[None, None] + sigmas[:, None, None, None] * noise_logits

        if b is not None:
            eps_b = torch.randn((S, int(K), b.numel()), device=W.device, dtype=W.dtype)
            logits = logits + sigmas[:, None, None, None] * (eps_b * b_scale[None, None]).unsqueeze(2)
        return logits

    # Backward-compatible aliases.
    def forward_lastlayer_mc_logits_from_features(self, feats, K=8, sigma=None):
        if sigma is None:
            sigma = torch.exp(torch.tensor(-5.0, device=feats.device, dtype=feats.dtype))
        return self.mc_logits_from_features(feats, sigma, K)

    def forward_lastlayer_mc_logits(self, x, K=8, sigma=None):
        return self.forward_lastlayer_mc_logits_from_features(self.extract_last_features(x), K, sigma)

    def forward_lastlayer_mc_logits_grid_from_features(self, feats, sigmas, K=8, same_noise_across_scales=True):
        return self.mc_logits_grid_from_features(feats, sigmas, K, same_noise_across_scales)

    @torch.no_grad()
    def posterior_std_summary(self) -> dict:
        out = {}
        if self.std_w is not None:
            out.update({
                "std_w_mean": float(self.std_w.mean()),
                "std_w_min": float(self.std_w.min()),
                "std_w_max": float(self.std_w.max()),
            })
        if self.std_b is not None:
            out.update({
                "std_b_mean": float(self.std_b.mean()),
                "std_b_min": float(self.std_b.min()),
                "std_b_max": float(self.std_b.max()),
            })
        if self.last_selected_sigma is not None:
            out["selected_sigma"] = self.last_selected_sigma
        out["posterior_scale_mode"] = str(getattr(self, "posterior_scale_mode", "scalar"))
        out["aniso_strength"] = float(getattr(self, "aniso_strength", 1.0))
        out["aniso_rank"] = int(getattr(self, "aniso_rank", 1))
        if self.last_aniso_w_factor is not None:
            f = self.last_aniso_w_factor.detach().float()
            out.update({
                "aniso_w_factor_mean": float(f.mean()),
                "aniso_w_factor_std": float(f.std(unbiased=False)),
                "aniso_w_factor_min": float(f.min()),
                "aniso_w_factor_max": float(f.max()),
                "aniso_w_factor_cv": float(f.std(unbiased=False) / f.mean().clamp_min(1e-12)),
            })
        if self.last_aniso_class_factor is not None:
            out["aniso_class_factor"] = self.last_aniso_class_factor.detach().cpu()
        if self.last_aniso_feature_factor is not None:
            out["aniso_feature_factor"] = self.last_aniso_feature_factor.detach().cpu()
        return out


# Compatibility name used by earlier experiments.
LaplaceWrapper_betta = LastLayerGaussianSampler
LaplaceWrapper = LastLayerGaussianSampler


# =============================================================================
# 3. BeTTA core: sigma selection, posterior prediction, trace logging
# =============================================================================


class BeTTACoreMixin:
    """Method-independent BeTTA logic.

    Any TTA method can reuse these functions:
      - Tent: use _betta_objective(...) over all samples
      - DeYO: use _betta_forward(...) for posterior prediction, then apply DeYO filters
      - ROID: use _betta_forward(...) for SLR/consistency/prior correction
    """

    def _init_betta_core(self, *, num_classes: int):
        self.K = int(os.environ.get("BETTA_NUM_MC", "300"))
        self.loss_type = env_str("BETTA_LOSS_TYPE", "C")
        self.selection = env_str("BETTA_SIGMA_SELECTION", "mi_elbow")
        self.fixed_sigma = float(os.environ.get("BETTA_FIXED_SIGMA", "1e-4"))
        self.same_noise = bool(int(os.environ.get("BETTA_SAME_NOISE", "1")))
        self.return_log_probs = bool(int(os.environ.get("BETTA_RETURN_LOG_PROBS", "1")))
        self.return_after_adapt = bool(int(os.environ.get("BETTA_RETURN_AFTER_ADAPT", "0")))
        self.force_bn1d_batch_stats = bool(int(os.environ.get("BETTA_FORCE_BN1D_BATCH_STATS", "0")))
        
        
        self.relative_floor = float(os.environ.get("BETTA_RELATIVE_FLOOR", "1e-12"))
        self.sigma_scale_mode = env_str("BETTA_SIGMA_SCALE_MODE", "he_in")
        self.bias_scale_mode = env_str("BETTA_BIAS_SCALE_MODE", "auto")

        # Selection-rule knobs. These are used by the label-free sigma selectors below.
        # Suffix forms such as mi_elbow_80 or agreement_90 override these defaults.
        self.mi_elbow_alpha = float(os.environ.get("BETTA_MI_ELBOW_ALPHA", "0.03"))
        self.agreement_threshold = float(os.environ.get("BETTA_AGREEMENT_THRESHOLD", "0.9"))
        self.he_norm_max = float(os.environ.get("BETTA_HE_NORM_MAX", "0.60"))

        self.sigma_interpolate = bool(int(os.environ.get("BETTA_SIGMA_INTERPOLATE", "1")))

        # Optional 2D/3D internal grid for factorized posterior scale.
        # When enabled, sigma selection searches over:
        #   sigma_grid x BETTA_ANISO_STRENGTH_GRID x BETTA_ANISO_RANK_GRID.
        # This is intentionally restricted to factorized posterior scale modes.
        self.factor_grid_enabled = bool(int(os.environ.get("BETTA_FACTOR_GRID", "0")))
        self.factor_strength_grid = env_float_list(
            "BETTA_ANISO_STRENGTH_GRID",
            env_float_list("BETTA_FACTOR_STRENGTH_GRID", [float(os.environ.get("BETTA_ANISO_STRENGTH", "1.0"))]),
        )
        self.factor_rank_grid = env_int_list(
            "BETTA_ANISO_RANK_GRID",
            env_int_list("BETTA_FACTOR_RANK_GRID", [int(float(os.environ.get("BETTA_ANISO_RANK", "1")))]),
        )

        # Optional loss knobs for MC/last-layer objectives.
        # These are intentionally environment-driven so run_swag.sh can grid them
        # without changing cfg files.
        self.loss_tau    = float(os.environ.get("BETTA_LOSS_TAU", "0.25"))
        self.loss_lambda = float(os.environ.get("BETTA_LOSS_LAMBDA", "1.0"))
        self.loss_gamma = float(os.environ.get("BETTA_LOSS_GAMMA", "1.0"))
        self.loss_margin = float(os.environ.get("BETTA_LOSS_MARGIN", "0.0"))
        self.loss_agreement_threshold = float(
            os.environ.get("BETTA_LOSS_AGREEMENT_THRESHOLD", str(self.agreement_threshold))
        )
        self.loss_detach_weights = bool(int(os.environ.get("BETTA_LOSS_DETACH_WEIGHTS", "1")))

        # Optional per-batch gradient-norm matching across BETTA_LOSS_TYPE
        # variants.  When enabled, the scalar objective is multiplied by a
        # detached factor so that its gradient w.r.t. the adapted parameters
        # has the requested L2 norm.  This removes pure loss-scale effects
        # (e.g. a/2a/3a/4a) while preserving each objective's gradient direction.
        self.grad_norm_match = bool(int(os.environ.get("BETTA_GRAD_NORM_MATCH", "0")))
        self.grad_norm_target = float(os.environ.get("BETTA_GRAD_NORM_TARGET", "1.0"))
        self.grad_norm_eps = float(os.environ.get("BETTA_GRAD_NORM_EPS", "1e-12"))
        self.grad_norm_max_scale = float(os.environ.get("BETTA_GRAD_NORM_MAX_SCALE", "1000000.0"))
        self.last_raw_grad_norm = float("nan")
        self.last_grad_norm_scale = 1.0

        self._num_classes_for_betta = int(num_classes)
        self.sigma_log = []
        self.scale_trace = []

    def _wrap_model_for_betta(self, model: nn.Module) -> LastLayerGaussianSampler:
        return LastLayerGaussianSampler(
            model,
            relative_floor=self.relative_floor,
            sigma_scale_mode=self.sigma_scale_mode,
            bias_scale_mode=self.bias_scale_mode,
            num_classes=self._num_classes_for_betta,
        )

    def _init_sigma_grid_after_super(self):
        W = self.model.last_layer.weight

        # Edit this list directly for quick experiments, or pass BETTA_SIGMA_GRID="1e-4,1e-3,...".
        grid_text = os.environ.get("BETTA_SIGMA_GRID", "").strip()
        if grid_text:
            values = [float(x.strip()) for x in grid_text.split(",") if x.strip()]
        else:
            values = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1.0, 2.0, 3.0, 4.0, 5.0]

        self.sigma_grid = torch.unique(
            torch.tensor(values, device=W.device, dtype=torch.float32).clamp_min(1e-12),
            sorted=True,
        )

        print(
            f"[BeTTA] K={self.K}, selection={self.selection}, loss_type={self.loss_type}, "
            f"scale={self.model.sigma_scale_mode}, bias={self.model.bias_scale_mode}, "
            f"posterior_scale={self.model.posterior_scale_mode}, "
            f"factor_grid={self.factor_grid_enabled}, "
            f"strength_grid={self.factor_strength_grid}, rank_grid={self.factor_rank_grid}, "
            f"grad_norm_match={self.grad_norm_match}, grad_norm_target={self.grad_norm_target}, "
            f"grid={self.sigma_grid.detach().cpu().tolist()}"
        )

    @staticmethod
    def _diff_wrt_log_sigma(y: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        if y.numel() == 1:
            return torch.zeros_like(y)

        x = sigmas.clamp_min(1e-12).log()
        d = torch.empty_like(y)
        d[0] = (y[1] - y[0]) / (x[1] - x[0]).clamp_min(1e-12)
        d[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2]).clamp_min(1e-12)
        if y.numel() > 2:
            d[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2]).clamp_min(1e-12)
        return d

    @staticmethod
    def _threshold_from_selection(selection: str, prefix: str, default: float) -> float:
        """Parse selectors like mi_elbow_80 / agreement_90 into 0.80 / 0.90."""
        sel = str(selection).lower().replace("-", "_")
        if not sel.startswith(prefix):
            return float(default)
        tail = sel[len(prefix):].strip("_")
        if not tail:
            return float(default)
        token = tail.split("_")[-1]
        try:
            value = float(token)
        except ValueError:
            return float(default)
        return value / 100.0 if value > 1.0 else value

    def _factorized_grid_active(self) -> bool:
        mode = str(getattr(self.model, "posterior_scale_mode", "scalar")).lower().replace("-", "_")
        return bool(getattr(self, "factor_grid_enabled", False)) and mode in {
            "factorized", "class_feature", "feature_class", "factorized_grid", "multi_grid", "rank_factorized"
        }

    def _set_factorized_state(self, strength: float, rank: int):
        self.model.posterior_scale_mode = "factorized"
        self.model.aniso_strength = float(strength)
        self.model.aniso_rank = int(max(1, rank))

    @torch.no_grad()
    def _select_factorized_sigma_grid_from_features(self, feats: torch.Tensor):
        """Select sigma, factorized anisotropy strength, and rank jointly.

        Searches over sigma_grid x BETTA_ANISO_STRENGTH_GRID x BETTA_ANISO_RANK_GRID.
        The returned sigma is still scalar, but the final MC forward uses the selected
        factorized per-weight posterior scale.
        """
        sigmas = self.sigma_grid.to(device=feats.device, dtype=feats.dtype).flatten()
        sel = str(self.selection).lower().replace("-", "_")
        n_sigma = int(sigmas.numel())

        strengths = [float(x) for x in getattr(self, "factor_strength_grid", [float(getattr(self.model, "aniso_strength", 1.0))])]
        ranks = [int(max(1, x)) for x in getattr(self, "factor_rank_grid", [int(getattr(self.model, "aniso_rank", 1))])]
        if len(strengths) == 0:
            strengths = [float(getattr(self.model, "aniso_strength", 1.0))]
        if len(ranks) == 0:
            ranks = [int(getattr(self.model, "aniso_rank", 1))]

        old_mode = str(getattr(self.model, "posterior_scale_mode", "scalar"))
        old_strength = float(getattr(self.model, "aniso_strength", 1.0))
        old_rank = int(getattr(self.model, "aniso_rank", 1))

        combo_strengths = []
        combo_ranks = []
        EH_rows = []
        HE_rows = []
        MI_rows = []
        joint_rows = []
        he_norm_rows = []
        agreement_rows = []
        dEH_rows = []
        dHE_rows = []
        dMI_rows = []

        try:
            for rank in ranks:
                for strength in strengths:
                    self._set_factorized_state(strength=strength, rank=rank)
                    logits_skbc = self.model.mc_logits_grid_from_features(
                        feats.detach(), sigmas, self.K, same_noise=self.same_noise
                    )
                    probs_skbc = logits_skbc.softmax(dim=-1)                         # [S,K,B,C]
                    p_mean_sbc = probs_skbc.mean(dim=1).clamp_min(1e-12)             # [S,B,C]
                    p_mean_sbc = p_mean_sbc / p_mean_sbc.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                    EH = entropy_from_probs(probs_skbc).mean(dim=(1, 2))             # [S]
                    HE = entropy_from_probs(p_mean_sbc).mean(dim=1)                  # [S]
                    MI = HE - EH
                    joint = HE + EH
                    he_norm = HE / math.log(max(int(self._num_classes_for_betta), 2))
                    dEH = self._diff_wrt_log_sigma(EH, sigmas)
                    dHE = self._diff_wrt_log_sigma(HE, sigmas)
                    dMI = self._diff_wrt_log_sigma(MI, sigmas)

                    mean_top1 = p_mean_sbc.argmax(dim=-1)                            # [S,B]
                    sample_top1 = probs_skbc.argmax(dim=-1)                          # [S,K,B]
                    agreement = (sample_top1 == mean_top1[:, None, :]).float().mean(dim=(1, 2))

                    combo_strengths.append(float(strength))
                    combo_ranks.append(int(rank))
                    EH_rows.append(EH)
                    HE_rows.append(HE)
                    MI_rows.append(MI)
                    joint_rows.append(joint)
                    he_norm_rows.append(he_norm)
                    agreement_rows.append(agreement)
                    dEH_rows.append(dEH)
                    dHE_rows.append(dHE)
                    dMI_rows.append(dMI)
        finally:
            # We set the selected state below. This restore protects exceptions.
            self.model.posterior_scale_mode = old_mode
            self.model.aniso_strength = old_strength
            self.model.aniso_rank = old_rank

        EH_gs = torch.stack(EH_rows, dim=0)                 # [G,S]
        HE_gs = torch.stack(HE_rows, dim=0)
        MI_gs = torch.stack(MI_rows, dim=0)
        joint_gs = torch.stack(joint_rows, dim=0)
        he_norm_gs = torch.stack(he_norm_rows, dim=0)
        agreement_gs = torch.stack(agreement_rows, dim=0)
        dEH_gs = torch.stack(dEH_rows, dim=0)
        dHE_gs = torch.stack(dHE_rows, dim=0)
        dMI_gs = torch.stack(dMI_rows, dim=0)

        G = int(EH_gs.shape[0])
        flat_shape = (G, n_sigma)

        def unravel(flat_idx: torch.Tensor | int) -> tuple[int, int]:
            idx = int(flat_idx.detach().item()) if torch.is_tensor(flat_idx) else int(flat_idx)
            return idx // n_sigma, idx % n_sigma

        def masked_flat_argmax(values: torch.Tensor, mask: torch.Tensor | None = None):
            v = values.reshape(-1)
            if mask is not None:
                m = mask.reshape(-1)
                if not bool(m.any().detach().item()):
                    return int(v.argmax().detach().item())
                vv = v.masked_fill(~m, torch.finfo(v.dtype).min)
                return int(vv.argmax().detach().item())
            return int(v.argmax().detach().item())

        def masked_flat_argmin(values: torch.Tensor, mask: torch.Tensor | None = None):
            v = values.reshape(-1)
            if mask is not None:
                m = mask.reshape(-1)
                if not bool(m.any().detach().item()):
                    return int(v.argmin().detach().item())
                vv = v.masked_fill(~m, torch.finfo(v.dtype).max)
                return int(vv.argmin().detach().item())
            return int(v.argmin().detach().item())

        # Joint selection rule over factor-combo x sigma grid.
        if sel == "fixed":
            fixed = torch.tensor(self.fixed_sigma, device=feats.device, dtype=feats.dtype).clamp(sigmas.min(), sigmas.max())
            s_idx = int(torch.argmin((sigmas - fixed).abs()).detach().item())
            # At fixed sigma, choose the factor setting with minimum HE+EH.
            g_idx = int(joint_gs[:, s_idx].argmin().detach().item())
            interp_kind = "factor_grid_fixed_sigma_min_joint"

        elif sel in {"max_dmi", "paper", "max_derivative", "max_dmi_linear", "max_dmi_quad"}:
            g_idx, s_idx = unravel(masked_flat_argmax(dMI_gs))
            interp_kind = "factor_grid_max_dmi"

        elif sel in {"max_mi", "max_epistemic", "max_mi_linear", "max_mi_quad"}:
            g_idx, s_idx = unravel(masked_flat_argmax(MI_gs))
            interp_kind = "factor_grid_max_mi"

        elif sel in {"min_joint", "min_he_eh", "min_joint_linear", "min_he_eh_linear", "min_joint_quad", "min_he_eh_quad"}:
            g_idx, s_idx = unravel(masked_flat_argmin(joint_gs))
            interp_kind = "factor_grid_min_joint"

        elif sel.startswith("agreement"):
            thr = self._threshold_from_selection(sel, "agreement", self.agreement_threshold)
            thr = float(min(max(thr, 0.0), 1.0))
            thr_t = torch.tensor(thr, device=agreement_gs.device, dtype=agreement_gs.dtype)
            cand_g = []
            cand_s = []
            cand_score = []
            for g in range(G):
                ok = torch.nonzero(agreement_gs[g] >= thr_t, as_tuple=False).flatten()
                s = int(ok[-1].detach().item()) if ok.numel() > 0 else int(agreement_gs[g].argmax().detach().item())
                cand_g.append(g); cand_s.append(s); cand_score.append(MI_gs[g, s] - 0.1 * joint_gs[g, s])
            pick = int(torch.stack(cand_score).argmax().detach().item())
            g_idx, s_idx = cand_g[pick], cand_s[pick]
            interp_kind = "factor_grid_agreement_boundary"

        elif sel.startswith("max_mi_agreement"):
            thr = self._threshold_from_selection(sel, "max_mi_agreement", self.agreement_threshold)
            thr = float(min(max(thr, 0.0), 1.0))
            mask = agreement_gs >= torch.tensor(thr, device=agreement_gs.device, dtype=agreement_gs.dtype)
            g_idx, s_idx = unravel(masked_flat_argmax(MI_gs, mask))
            interp_kind = "factor_grid_max_mi_agreement"

        elif sel in {"entropy_guarded_max_mi", "he_guarded_max_mi"}:
            mask = he_norm_gs <= float(self.he_norm_max)
            g_idx, s_idx = unravel(masked_flat_argmax(MI_gs, mask))
            interp_kind = "factor_grid_entropy_guarded_max_mi"

        elif sel in {"entropy_guarded_max_dmi", "he_guarded_max_dmi"}:
            mask = he_norm_gs <= float(self.he_norm_max)
            g_idx, s_idx = unravel(masked_flat_argmax(dMI_gs, mask))
            interp_kind = "factor_grid_entropy_guarded_max_dmi"

        elif sel.startswith("mi_elbow"):
            # For each factor setting, pick the first sigma reaching the MI elbow;
            # then choose the candidate with the best unsupervised joint entropy.
            alpha = self._threshold_from_selection(sel, "mi_elbow", self.mi_elbow_alpha)
            alpha = float(min(max(alpha, 0.0), 1.0))
            use_agreement = "agreement" in sel
            thr = self._threshold_from_selection(sel, "mi_elbow_agreement", self.agreement_threshold)
            thr = float(min(max(thr, 0.0), 1.0))
            thr_t = torch.tensor(thr, device=agreement_gs.device, dtype=agreement_gs.dtype)

            cand_g = []
            cand_s = []
            cand_score = []
            for g in range(G):
                target = MI_gs[g].min() + alpha * (MI_gs[g].max() - MI_gs[g].min())
                mask = MI_gs[g] >= target
                if use_agreement:
                    mask = mask & (agreement_gs[g] >= thr_t)
                ids = torch.nonzero(mask, as_tuple=False).flatten()
                s = int(ids[0].detach().item()) if ids.numel() > 0 else int(MI_gs[g].argmax().detach().item())
                cand_g.append(g); cand_s.append(s); cand_score.append(joint_gs[g, s])
            pick = int(torch.stack(cand_score).argmin().detach().item())
            g_idx, s_idx = cand_g[pick], cand_s[pick]
            interp_kind = "factor_grid_mi_elbow" if not use_agreement else "factor_grid_mi_elbow_agreement"

        else:
            raise ValueError(
                f"Unknown BETTA_SIGMA_SELECTION={self.selection} for factorized grid. "
                "Use fixed, max_dmi, max_mi, min_joint, mi_elbow, agreement, "
                "max_mi_agreement, entropy_guarded_max_mi, or entropy_guarded_max_dmi."
            )

        sigma = sigmas[int(s_idx)]
        selected_strength = float(combo_strengths[int(g_idx)])
        selected_rank = int(combo_ranks[int(g_idx)])
        self._set_factorized_state(strength=selected_strength, rank=selected_rank)

        stats = {
            "idx": int(s_idx),
            "idx_anchor": int(s_idx),
            "idx_nearest": int(s_idx),
            "sigma": float(sigma.detach().item()),
            "sigma_grid_value": float(sigmas[int(s_idx)].detach().item()),
            "sigma_grid_value_anchor": float(sigmas[int(s_idx)].detach().item()),
            "sigma_grid_value_nearest": float(sigmas[int(s_idx)].detach().item()),
            "sigma_fractional_idx": float(s_idx),
            "sigma_interpolated": False,
            "sigma_interp_kind": interp_kind,
            "boundary": bool(int(s_idx) == 0 or int(s_idx) == n_sigma - 1),
            "factor_grid_enabled": True,
            "factor_combo_idx": int(g_idx),
            "factor_selected_strength": selected_strength,
            "factor_selected_rank": selected_rank,
            "factor_strength_grid": torch.tensor(strengths, dtype=torch.float32).cpu(),
            "factor_rank_grid": torch.tensor(ranks, dtype=torch.int64).cpu(),
            "factor_combo_strength": torch.tensor(combo_strengths, dtype=torch.float32).cpu(),
            "factor_combo_rank": torch.tensor(combo_ranks, dtype=torch.int64).cpu(),
            "factor_grid_shape": torch.tensor([G, n_sigma], dtype=torch.int64).cpu(),

            # Backward-compatible 1D curves for the selected factor combo.
            "EH": EH_gs[g_idx].detach().cpu(),
            "HE": HE_gs[g_idx].detach().cpu(),
            "MI": MI_gs[g_idx].detach().cpu(),
            "joint": joint_gs[g_idx].detach().cpu(),
            "HE_norm": he_norm_gs[g_idx].detach().cpu(),
            "agreement": agreement_gs[g_idx].detach().cpu(),
            "dEH": dEH_gs[g_idx].detach().cpu(),
            "dHE": dHE_gs[g_idx].detach().cpu(),
            "dMI": dMI_gs[g_idx].detach().cpu(),

            # Full factor-combo x sigma grids, flattened for trace serialization.
            "factor_grid_EH": EH_gs.detach().cpu().reshape(-1),
            "factor_grid_HE": HE_gs.detach().cpu().reshape(-1),
            "factor_grid_MI": MI_gs.detach().cpu().reshape(-1),
            "factor_grid_joint": joint_gs.detach().cpu().reshape(-1),
            "factor_grid_HE_norm": he_norm_gs.detach().cpu().reshape(-1),
            "factor_grid_agreement": agreement_gs.detach().cpu().reshape(-1),
            "factor_grid_dEH": dEH_gs.detach().cpu().reshape(-1),
            "factor_grid_dHE": dHE_gs.detach().cpu().reshape(-1),
            "factor_grid_dMI": dMI_gs.detach().cpu().reshape(-1),
        }
        return sigma, stats

    @torch.no_grad()
    def _select_sigma_from_features(self, feats: torch.Tensor):
        """Select one sigma for the current batch. No labels, no gradients.

        The grid is still evaluated discretely.  Continuous sigma values are
        produced only by cheap interpolation of scalar curves in log-sigma space.

        Important detail:
          - The exact min/max of a piecewise-linear interpolant is normally at a
            grid knot.  Therefore *_linear selectors mostly return grid values.
          - Use *_quad selectors when you want a local continuous extremum
            estimate around the best grid point.
          - mi_elbow / agreement selectors use linear threshold crossing and can
            naturally return continuous values.

        Supported BETTA_SIGMA_SELECTION options:
          fixed
          max_dmi / paper / max_derivative
          max_dmi_linear / max_dmi_quad
          max_mi / max_epistemic
          max_mi_linear / max_mi_quad
          min_joint / min_he_eh
          min_joint_linear / min_joint_quad
          mi_elbow / mi_elbow_80 / mi_elbow_90
          agreement / agreement_85 / agreement_90
          max_mi_agreement / max_mi_agreement_85
          entropy_guarded_max_mi / entropy_guarded_max_dmi
          mi_elbow_agreement / mi_elbow_agreement_80 / mi_elbow_agreement_90
        """
        sigmas = self.sigma_grid.to(device=feats.device, dtype=feats.dtype).flatten()
        sel = str(self.selection).lower().replace("-", "_")
        n_grid = int(sigmas.numel())

        if self._factorized_grid_active():
            return self._select_factorized_sigma_grid_from_features(feats)

        def _float_scalar(x) -> float:
            if torch.is_tensor(x):
                return float(x.detach().item())
            return float(x)

        def _nearest_grid_idx(sigma_value: torch.Tensor) -> int:
            sigma_value = torch.as_tensor(sigma_value, device=sigmas.device, dtype=sigmas.dtype)
            return int(torch.argmin((sigmas - sigma_value).abs()).detach().item())

        def _fractional_grid_idx(sigma_value: torch.Tensor) -> float:
            """Map a continuous sigma to a fractional index on the log-sigma grid."""
            sigma_value = torch.as_tensor(sigma_value, device=sigmas.device, dtype=sigmas.dtype)
            s = sigma_value.clamp(sigmas.min(), sigmas.max())
            x_log_local = sigmas.clamp_min(1e-12).log()
            sx = s.clamp_min(1e-12).log()

            if bool((sx <= x_log_local[0]).detach().item()):
                return 0.0
            if bool((sx >= x_log_local[-1]).detach().item()):
                return float(n_grid - 1)

            right = int(torch.searchsorted(x_log_local, sx, right=False).detach().item())
            right = min(max(right, 1), n_grid - 1)
            left = right - 1
            denom = (x_log_local[right] - x_log_local[left]).clamp_min(1e-12)
            t = ((sx - x_log_local[left]) / denom).clamp(0.0, 1.0)
            return float(left + _float_scalar(t))

        def _pack_stats(
            *,
            sigma_value: torch.Tensor,
            idx_anchor: int,
            interp_kind: str,
            extra: dict | None = None,
        ) -> dict:
            sigma_value = torch.as_tensor(sigma_value, device=feats.device, dtype=feats.dtype).clamp(
                sigmas.min(), sigmas.max()
            )
            idx_anchor = int(min(max(idx_anchor, 0), n_grid - 1))
            idx_nearest = _nearest_grid_idx(sigma_value)
            frac_idx = _fractional_grid_idx(sigma_value)
            selected = _float_scalar(sigma_value)
            anchor_value = _float_scalar(sigmas[idx_anchor])
            nearest_value = _float_scalar(sigmas[idx_nearest])

            stats = {
                # Backward-compatible diagnostic anchor.
                "idx": idx_anchor,

                # Explicit diagnostics for continuous sigma selection.
                "idx_anchor": idx_anchor,
                "idx_nearest": idx_nearest,
                "sigma": selected,
                "sigma_grid_value": anchor_value,
                "sigma_grid_value_anchor": anchor_value,
                "sigma_grid_value_nearest": nearest_value,
                "sigma_fractional_idx": float(frac_idx),
                "sigma_interpolated": bool(abs(selected - anchor_value) > 1e-12),
                "sigma_interp_kind": str(interp_kind),
                "boundary": bool(idx_anchor == 0 or idx_anchor == n_grid - 1),
            }
            if extra:
                stats.update(extra)
            return stats

        if sel == "fixed" or n_grid == 1:
            sigma = (
                torch.tensor(self.fixed_sigma, device=feats.device, dtype=feats.dtype)
                if sel == "fixed"
                else sigmas[0]
            ).clamp(sigmas.min(), sigmas.max())
            idx = _nearest_grid_idx(sigma)
            return sigma, _pack_stats(
                sigma_value=sigma,
                idx_anchor=idx,
                interp_kind="fixed_or_single",
            )

        logits_skbc = self.model.mc_logits_grid_from_features(
            feats.detach(), sigmas, self.K, same_noise=self.same_noise
        )
        probs_skbc = logits_skbc.softmax(dim=-1)                         # [S,K,B,C]
        p_mean_sbc = probs_skbc.mean(dim=1).clamp_min(1e-12)             # [S,B,C]
        p_mean_sbc = p_mean_sbc / p_mean_sbc.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        EH = entropy_from_probs(probs_skbc).mean(dim=(1, 2))             # [S]
        HE = entropy_from_probs(p_mean_sbc).mean(dim=1)                  # [S]
        MI = HE - EH
        joint = HE + EH
        he_norm = HE / math.log(max(int(self._num_classes_for_betta), 2))

        dEH = self._diff_wrt_log_sigma(EH, sigmas)
        dHE = self._diff_wrt_log_sigma(HE, sigmas)
        dMI = self._diff_wrt_log_sigma(MI, sigmas)

        # Agreement: how often MC-sampled top-1 agrees with posterior-mean top-1.
        # High agreement means the perturbation has not destroyed the predicted class.
        mean_top1 = p_mean_sbc.argmax(dim=-1)                            # [S,B]
        sample_top1 = probs_skbc.argmax(dim=-1)                          # [S,K,B]
        agreement = (sample_top1 == mean_top1[:, None, :]).float().mean(dim=(1, 2))  # [S]

        x_log = sigmas.clamp_min(1e-12).log()

        def _first_true(mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            ids = torch.nonzero(mask, as_tuple=False).flatten()
            return ids[0] if ids.numel() > 0 else fallback

        def _last_true(mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            ids = torch.nonzero(mask, as_tuple=False).flatten()
            return ids[-1] if ids.numel() > 0 else fallback

        def _argmax_masked(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            ids = torch.nonzero(mask, as_tuple=False).flatten()
            if ids.numel() == 0:
                return fallback
            return ids[values[ids].argmax()]

        def _argmin_masked(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            ids = torch.nonzero(mask, as_tuple=False).flatten()
            if ids.numel() == 0:
                return fallback
            return ids[values[ids].argmin()]

        def _interp_level_sigma(
            y: torch.Tensor,
            level: torch.Tensor | float,
            left_idx: int,
            right_idx: int,
        ) -> tuple[torch.Tensor, bool]:
            """Linear interpolation of y(x) crossing in log-sigma space."""
            n = int(y.numel())
            if left_idx < 0 or right_idx < 0 or left_idx >= n or right_idx >= n:
                j = min(max(right_idx, 0), n - 1)
                return sigmas[j], False

            y0, y1 = y[left_idx], y[right_idx]
            x0, x1 = x_log[left_idx], x_log[right_idx]
            level_t = torch.as_tensor(level, device=y.device, dtype=y.dtype)

            denom = y1 - y0
            if torch.abs(denom).detach().item() < 1e-12:
                return sigmas[right_idx], False

            t = ((level_t - y0) / denom).clamp(0.0, 1.0)
            x_hat = x0 + t * (x1 - x0)
            sigma_hat = x_hat.exp().clamp(sigmas.min(), sigmas.max())
            return sigma_hat.to(device=sigmas.device, dtype=sigmas.dtype), True

        def _linear_spline_extreme_sigma(
            y: torch.Tensor,
            *,
            maximize: bool,
            valid_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, int, bool]:
            """Exact extremum of a piecewise-linear interpolant.

            For a piecewise-linear curve, the min/max is at a grid knot unless
            there is a flat optimal segment.  If such a segment exists, return
            the log-space midpoint of the longest optimal flat block.
            """
            if valid_mask is None:
                valid_mask = torch.ones_like(y, dtype=torch.bool)

            ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
            if ids.numel() == 0:
                idx = int((y.argmax() if maximize else y.argmin()).detach().item())
                return sigmas[idx], idx, False

            vals = y[ids]
            best_val = vals.max() if maximize else vals.min()
            eps = max(abs(_float_scalar(best_val)) * 1e-7, 1e-12)
            best_ids = ids[torch.abs(y[ids] - best_val) <= eps]

            if best_ids.numel() == 0:
                idx_t = ids[vals.argmax()] if maximize else ids[vals.argmin()]
                idx = int(idx_t.detach().item())
                return sigmas[idx], idx, False

            best_ids_sorted = best_ids.sort().values
            blocks: list[tuple[int, int]] = []
            start = int(best_ids_sorted[0].detach().item())
            prev = start
            for z in best_ids_sorted[1:]:
                z_int = int(z.detach().item())
                if z_int == prev + 1:
                    prev = z_int
                else:
                    blocks.append((start, prev))
                    start = z_int
                    prev = z_int
            blocks.append((start, prev))

            left, right = max(blocks, key=lambda p: p[1] - p[0])
            if right > left:
                x_mid = 0.5 * (x_log[left] + x_log[right])
                sigma_hat = x_mid.exp().clamp(sigmas.min(), sigmas.max())
                anchor = int(round(0.5 * (left + right)))
                return sigma_hat.to(device=sigmas.device, dtype=sigmas.dtype), anchor, True

            idx = left
            return sigmas[idx], idx, False

        def _quad_vertex_sigma(
            y: torch.Tensor,
            idx: int,
            *,
            maximize: bool = True,
            valid_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, bool]:
            """Local quadratic interpolation around idx in log-sigma space.

            Fits a parabola through idx-1, idx, idx+1 and returns its vertex.
            Used only when the vertex remains inside the local interval and the
            curvature matches the requested extremum.
            """
            n = int(y.numel())
            if idx <= 0 or idx >= n - 1:
                return sigmas[idx], False

            if valid_mask is not None:
                local_ok = (
                    bool(valid_mask[idx - 1].detach().item())
                    and bool(valid_mask[idx].detach().item())
                    and bool(valid_mask[idx + 1].detach().item())
                )
                if not local_ok:
                    return sigmas[idx], False

            x0, x1, x2 = x_log[idx - 1], x_log[idx], x_log[idx + 1]
            y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]

            denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
            if torch.abs(denom).detach().item() < 1e-12:
                return sigmas[idx], False

            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            b = (x2.pow(2) * (y0 - y1) + x1.pow(2) * (y2 - y0) + x0.pow(2) * (y1 - y2)) / denom

            if torch.abs(a).detach().item() < 1e-12:
                return sigmas[idx], False

            # For max: a < 0. For min: a > 0.
            a_val = _float_scalar(a)
            if maximize and a_val >= 0.0:
                return sigmas[idx], False
            if (not maximize) and a_val <= 0.0:
                return sigmas[idx], False

            xv = -b / (2.0 * a)
            lo = torch.minimum(x0, x2)
            hi = torch.maximum(x0, x2)

            if bool((xv < lo).detach().item()) or bool((xv > hi).detach().item()):
                return sigmas[idx], False

            sigma_hat = xv.exp().clamp(sigmas.min(), sigmas.max())
            return sigma_hat.to(device=sigmas.device, dtype=sigmas.dtype), True

        interpolate = bool(getattr(self, "sigma_interpolate", True))
        interp_kind = "none"

        # ----------------------------------------------------------------------
        # Selection rules.
        # ----------------------------------------------------------------------
        if sel in {"max_dmi_linear"}:
            sigma_hat, idx_int, ok = _linear_spline_extreme_sigma(dMI, maximize=True)
            interp_kind = "linear_spline_dmi_max_flat_midpoint" if ok else "linear_spline_dmi_max_grid"

        elif sel in {"max_dmi_quad"}:
            idx_int = int(dMI.argmax().detach().item())
            sigma_hat, ok = _quad_vertex_sigma(dMI, idx_int, maximize=True)
            interp_kind = "quadratic_logsigma_dmi_max" if ok else "discrete_dmi_max"

        elif sel in {"max_dmi", "paper", "max_derivative"}:
            idx_int = int(dMI.argmax().detach().item())
            sigma_hat = sigmas[idx_int]
            if interpolate:
                sigma_hat, ok = _quad_vertex_sigma(dMI, idx_int, maximize=True)
                interp_kind = "quadratic_logsigma_dmi_max" if ok else "discrete_dmi_max"
            else:
                interp_kind = "discrete_dmi_max"

        elif sel in {"max_mi_linear", "max_epistemic_linear"}:
            sigma_hat, idx_int, ok = _linear_spline_extreme_sigma(MI, maximize=True)
            interp_kind = "linear_spline_mi_max_flat_midpoint" if ok else "linear_spline_mi_max_grid"

        elif sel in {"max_mi_quad", "max_epistemic_quad"}:
            idx_int = int(MI.argmax().detach().item())
            sigma_hat, ok = _quad_vertex_sigma(MI, idx_int, maximize=True)
            interp_kind = "quadratic_logsigma_mi_max" if ok else "discrete_mi_max"

        elif sel in {"max_mi", "max_epistemic"}:
            idx_int = int(MI.argmax().detach().item())
            sigma_hat = sigmas[idx_int]
            if interpolate:
                sigma_hat, ok = _quad_vertex_sigma(MI, idx_int, maximize=True)
                interp_kind = "quadratic_logsigma_mi_max" if ok else "discrete_mi_max"
            else:
                interp_kind = "discrete_mi_max"

        elif sel in {"min_joint_linear", "min_he_eh_linear"}:
            sigma_hat, idx_int, ok = _linear_spline_extreme_sigma(joint, maximize=False)
            interp_kind = "linear_spline_joint_min_flat_midpoint" if ok else "linear_spline_joint_min_grid"

        elif sel in {"min_joint_quad", "min_he_eh_quad"}:
            idx_int = int(joint.argmin().detach().item())
            sigma_hat, ok = _quad_vertex_sigma(joint, idx_int, maximize=False)
            interp_kind = "quadratic_logsigma_joint_min" if ok else "discrete_joint_min"

        elif sel in {"min_joint", "min_he_eh"}:
            idx_int = int(joint.argmin().detach().item())
            sigma_hat = sigmas[idx_int]
            if interpolate:
                sigma_hat, ok = _quad_vertex_sigma(joint, idx_int, maximize=False)
                interp_kind = "quadratic_logsigma_joint_min" if ok else "discrete_joint_min"
            else:
                interp_kind = "discrete_joint_min"

        elif sel.startswith("mi_elbow") and "agreement" not in sel:
            # Smallest sigma that reaches alpha of the observed MI range.
            alpha = self._threshold_from_selection(sel, "mi_elbow", self.mi_elbow_alpha)
            alpha = float(min(max(alpha, 0.0), 1.0))
            target = MI.min() + alpha * (MI.max() - MI.min())

            idx = _first_true(MI >= target, MI.argmax())
            idx_int = int(idx.detach().item())
            sigma_hat = sigmas[idx_int]

            if interpolate and idx_int > 0:
                prev = idx_int - 1
                if bool((MI[prev] < target).detach().item()):
                    sigma_hat, ok = _interp_level_sigma(MI, target, prev, idx_int)
                    interp_kind = "linear_logsigma_mi_elbow" if ok else "discrete_mi_elbow"
                else:
                    interp_kind = "discrete_mi_elbow"
            else:
                interp_kind = "discrete_mi_elbow"

        elif sel.startswith("agreement"):
            # Largest sigma that keeps MC top-1 agreement above threshold.
            thr = self._threshold_from_selection(sel, "agreement", self.agreement_threshold)
            thr = float(min(max(thr, 0.0), 1.0))
            thr_t = torch.tensor(thr, device=agreement.device, dtype=agreement.dtype)

            idx = _last_true(agreement >= thr_t, agreement.argmax())
            idx_int = int(idx.detach().item())
            sigma_hat = sigmas[idx_int]

            if interpolate and idx_int < n_grid - 1:
                nxt = idx_int + 1
                if bool((agreement[idx_int] >= thr_t).detach().item()) and bool((agreement[nxt] < thr_t).detach().item()):
                    sigma_hat, ok = _interp_level_sigma(agreement, thr_t, idx_int, nxt)
                    interp_kind = "linear_logsigma_agreement_boundary" if ok else "discrete_agreement"
                else:
                    interp_kind = "discrete_agreement"
            else:
                interp_kind = "discrete_agreement"

        elif sel.startswith("max_mi_agreement"):
            # Maximize MI, but only among sigmas whose MC samples retain top-1 agreement.
            thr = self._threshold_from_selection(sel, "max_mi_agreement", self.agreement_threshold)
            thr = float(min(max(thr, 0.0), 1.0))
            thr_t = torch.tensor(thr, device=agreement.device, dtype=agreement.dtype)
            mask = agreement >= thr_t

            idx = _argmax_masked(MI, mask, MI.argmax())
            idx_int = int(idx.detach().item())
            sigma_hat = sigmas[idx_int]

            if interpolate:
                sigma_hat, ok = _quad_vertex_sigma(MI, idx_int, maximize=True, valid_mask=mask)
                interp_kind = "quadratic_logsigma_mi_max_agreement" if ok else "discrete_max_mi_agreement"
            else:
                interp_kind = "discrete_max_mi_agreement"

        elif sel in {"entropy_guarded_max_mi", "he_guarded_max_mi"}:
            # Maximize MI without allowing posterior-mean entropy to become too uniform.
            mask = he_norm <= float(self.he_norm_max)
            idx = _argmax_masked(MI, mask, he_norm.argmin())
            idx_int = int(idx.detach().item())
            sigma_hat = sigmas[idx_int]

            if interpolate:
                sigma_hat, ok = _quad_vertex_sigma(MI, idx_int, maximize=True, valid_mask=mask)
                interp_kind = "quadratic_logsigma_entropy_guarded_mi_max" if ok else "discrete_entropy_guarded_max_mi"
            else:
                interp_kind = "discrete_entropy_guarded_max_mi"

        elif sel in {"entropy_guarded_max_dmi", "he_guarded_max_dmi"}:
            mask = he_norm <= float(self.he_norm_max)
            idx = _argmax_masked(dMI, mask, dMI.argmax())
            idx_int = int(idx.detach().item())
            sigma_hat = sigmas[idx_int]

            if interpolate:
                sigma_hat, ok = _quad_vertex_sigma(dMI, idx_int, maximize=True, valid_mask=mask)
                interp_kind = "quadratic_logsigma_entropy_guarded_dmi_max" if ok else "discrete_entropy_guarded_max_dmi"
            else:
                interp_kind = "discrete_entropy_guarded_max_dmi"

        elif sel.startswith("mi_elbow_agreement"):
            # Enough MI, minimal sigma, while preserving top-1 agreement.
            thr = self._threshold_from_selection(sel, "mi_elbow_agreement", self.agreement_threshold)
            thr = float(min(max(thr, 0.0), 1.0))
            thr_t = torch.tensor(thr, device=agreement.device, dtype=agreement.dtype)

            alpha = float(min(max(self.mi_elbow_alpha, 0.0), 1.0))
            target = MI.min() + alpha * (MI.max() - MI.min())

            mi_mask = MI >= target
            ag_mask = agreement >= thr_t
            both_mask = mi_mask & ag_mask

            idx = _first_true(both_mask, _first_true(mi_mask, MI.argmax()))
            idx_int = int(idx.detach().item())
            sigma_hat = sigmas[idx_int]

            if interpolate and idx_int > 0:
                prev = idx_int - 1
                can_interp = (
                    bool((MI[prev] < target).detach().item())
                    and bool((agreement[prev] >= thr_t).detach().item())
                    and bool((agreement[idx_int] >= thr_t).detach().item())
                )

                if can_interp:
                    sigma_hat, ok = _interp_level_sigma(MI, target, prev, idx_int)
                    interp_kind = "linear_logsigma_mi_elbow_agreement" if ok else "discrete_mi_elbow_agreement"
                else:
                    interp_kind = "discrete_mi_elbow_agreement"
            else:
                interp_kind = "discrete_mi_elbow_agreement"

        else:
            raise ValueError(
                f"Unknown BETTA_SIGMA_SELECTION={self.selection}. Supported: "
                "fixed, max_dmi/paper/max_derivative, max_dmi_linear, max_dmi_quad, "
                "max_mi/max_epistemic, max_mi_linear, max_mi_quad, "
                "min_joint/min_he_eh, min_joint_linear, min_joint_quad, "
                "mi_elbow[_80/_90], agreement[_85/_90], max_mi_agreement[_85/_90], "
                "entropy_guarded_max_mi, entropy_guarded_max_dmi, "
                "mi_elbow_agreement[_80/_90]."
            )

        sigma = torch.as_tensor(sigma_hat, device=feats.device, dtype=feats.dtype).clamp(sigmas.min(), sigmas.max())
        stats = _pack_stats(
            sigma_value=sigma,
            idx_anchor=int(idx_int),
            interp_kind=interp_kind,
            extra={
                "EH": EH.detach().cpu(),
                "HE": HE.detach().cpu(),
                "MI": MI.detach().cpu(),
                "joint": joint.detach().cpu(),
                "HE_norm": he_norm.detach().cpu(),
                "agreement": agreement.detach().cpu(),
                "dEH": dEH.detach().cpu(),
                "dHE": dHE.detach().cpu(),
                "dMI": dMI.detach().cpu(),
            },
        )
        return sigma, stats

    def _posterior_from_mc_logits(self, logits_kbc: torch.Tensor):
        """Compute posterior predictive and MC statistics.

        Args:
            logits_kbc: MC logits with shape [K, B, C].

        Returns:
            output: log(p_mean) by default, suitable for downstream softmax-style metrics.
            p_mean: posterior predictive mean, shape [B, C].
            HE: entropy of p_mean, shape [B].
            EH: mean entropy of MC samples, shape [B].
            probs_kbc: MC probabilities, shape [K, B, C].
        """
        probs_kbc = logits_kbc.softmax(dim=-1)
        p_mean = probs_kbc.mean(dim=0).clamp_min(1e-12)
        p_mean = p_mean / p_mean.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        EH = entropy_from_probs(probs_kbc).mean(dim=0)  # [B]
        HE = entropy_from_probs(p_mean)                 # [B]
        output = p_mean.log() if self.return_log_probs else p_mean
        return output, p_mean, HE, EH, probs_kbc

    def _betta_forward(self, imgs: torch.Tensor, sigma=None, *, select_if_none: bool = True):
        """One BeTTA posterior prediction pass.

        If sigma is None, choose sigma from current batch features.
        If sigma is provided, reuse it. This is useful for consistency branches.
        """
        feats = self.model.extract_last_features(imgs)

        if sigma is None and select_if_none:
            sigma, stats = self._select_sigma_from_features(feats)
        elif sigma is None:
            sigma = self.sigma_grid[0].to(device=feats.device, dtype=feats.dtype)
            stats = {"idx": 0, "sigma": float(sigma)}
        else:
            sigma = torch.as_tensor(sigma, device=feats.device, dtype=feats.dtype)
            grid = self.sigma_grid.to(feats.device, feats.dtype)
            idx = int(torch.argmin((grid - sigma).abs()))
            stats = {
                "idx": idx,
                "idx_anchor": idx,
                "idx_nearest": idx,
                "sigma": float(sigma.detach().item()),
                "sigma_grid_value": float(grid[idx].detach().item()),
                "sigma_grid_value_anchor": float(grid[idx].detach().item()),
                "sigma_grid_value_nearest": float(grid[idx].detach().item()),
                "sigma_fractional_idx": float(idx),
                "sigma_interpolated": bool(abs(float(sigma.detach().item()) - float(grid[idx].detach().item())) > 1e-12),
                "sigma_interp_kind": "provided",
                "boundary": bool(idx == 0 or idx == grid.numel() - 1),
            }

        logits_kbc = self.model.mc_logits_from_features(feats, sigma, self.K)
        output, p_mean, HE, EH, probs_kbc = self._posterior_from_mc_logits(logits_kbc)

        # Deterministic last-layer logits at the selected current features.
        # This lets optional losses use the last-layer geometry without another full forward.
        base_logits = F.linear(feats, self.model.last_layer.weight, self.model.last_layer.bias)

        # Per-sample MC diagnostics used by the optional losses.
        mean_top1 = p_mean.argmax(dim=-1)                                  # [B]
        sample_top1 = probs_kbc.argmax(dim=-1)                             # [K, B]
        agreement = (sample_top1 == mean_top1.unsqueeze(0)).float().mean(dim=0)
        mc_var = probs_kbc.var(dim=0, unbiased=False).sum(dim=-1)           # [B]
        logit_var = logits_kbc.var(dim=0, unbiased=False).mean(dim=-1)      # [B]

        out = {
            "output": output,
            "p_mean": p_mean,
            "HE": HE,
            "EH": EH,
            "MI": HE - EH,
            "sigma": sigma,
            "stats": stats,
            "features": feats,
            "logits_kbc": logits_kbc,
            "probs_kbc": probs_kbc,
            "base_logits": base_logits,
            "agreement": agreement,
            "mc_var": mc_var,
            "logit_var": logit_var,
        }
        # Keep backward compatibility with callers that still invoke
        # _betta_objective(HE, EH) instead of _betta_objective(betta_out).
        self._last_betta_out = out
        return out

    def _safe_tau(self) -> float:
        return max(float(getattr(self, "loss_tau", 0.25)), 1e-6)

    def _match_betta_gradient_norm(self, loss: torch.Tensor) -> torch.Tensor:
        """Rescale a scalar loss to a fixed adapted-parameter gradient norm.

        The scale factor is detached, so this changes only gradient magnitude,
        not gradient direction.  With matching enabled, objectives that differ
        only by a positive scalar multiplier (a/2a/3a/4a, for example) produce
        the same optimizer gradient up to floating-point roundoff.
        """
        return self._match_gradient_norm(loss)

    def _safe_log(self, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return p.clamp_min(eps).log()

    def _normalize_loss_weights(self, w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Normalize per-sample weights to mean 1 to reduce LR confounding."""
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        w = w.clamp_min(0.0)
        if bool(getattr(self, "loss_detach_weights", True)):
            w = w.detach()
        return w / w.mean().clamp_min(eps)

    def _low_score_weight(self, score: torch.Tensor, tau: float | None = None) -> torch.Tensor:
        """High weight for samples whose score is below the batch center."""
        tau = self._safe_tau() if tau is None else max(float(tau), 1e-6)
        z = ((score - score.detach().mean()) / tau).clamp(-20.0, 20.0)
        return self._normalize_loss_weights(torch.exp(-z))

    def _high_score_weight(self, score: torch.Tensor, tau: float | None = None) -> torch.Tensor:
        """High weight for samples whose score is above the batch center."""
        tau = self._safe_tau() if tau is None else max(float(tau), 1e-6)
        z = ((score - score.detach().mean()) / tau).clamp(-20.0, 20.0)
        return self._normalize_loss_weights(torch.exp(z))

    def _agreement_weight(self, agreement: torch.Tensor) -> torch.Tensor:
        """Soft gate around BETTA_LOSS_AGREEMENT_THRESHOLD."""
        tau = self._safe_tau()
        thr = float(getattr(self, "loss_agreement_threshold", 0.99))
        w = torch.sigmoid((agreement - thr) / tau)
        return self._normalize_loss_weights(w)

    def _pseudo_ce_from_pmean(self, p_mean: torch.Tensor) -> torch.Tensor:
        y_hat = p_mean.detach().argmax(dim=-1)
        return F.nll_loss(self._safe_log(p_mean), y_hat, reduction="none")

    def _sharpened_ce_from_pmean(self, p_mean: torch.Tensor) -> torch.Tensor:
        # tau < 1 sharpens; with default BETTA_LOSS_TAU=0.25 this is aggressive.
        tau = self._safe_tau()
        q = p_mean.detach().clamp_min(1e-12).pow(1.0 / tau)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return -(q * self._safe_log(p_mean)).sum(dim=-1)

    def _kl_pmean_to_samples(self, p_mean: torch.Tensor, probs_kbc: torch.Tensor) -> torch.Tensor:
        """Per-sample mean_k KL(stopgrad(p_mean) || p_k), shape [B]."""
        teacher = p_mean.detach().clamp_min(1e-12)
        return (teacher.unsqueeze(0) * (teacher.log().unsqueeze(0) - self._safe_log(probs_kbc))).sum(dim=-1).mean(dim=0)

    def _kl_samples_to_pmean(self, p_mean: torch.Tensor, probs_kbc: torch.Tensor) -> torch.Tensor:
        """Per-sample mean_k KL(p_k || stopgrad(p_mean)), shape [B]."""
        teacher = p_mean.detach().clamp_min(1e-12)
        return (probs_kbc.clamp_min(1e-12) * (self._safe_log(probs_kbc) - teacher.log().unsqueeze(0))).sum(dim=-1).mean(dim=0)

    def _mc_margin_loss(self, logits_kbc: torch.Tensor, p_mean: torch.Tensor) -> torch.Tensor:
        """Hinge-like loss for keeping the posterior top-1 logit above competitors under MC noise."""
        K, B, C = logits_kbc.shape
        y_hat = p_mean.detach().argmax(dim=-1)  # [B]
        y_exp = y_hat.unsqueeze(0).expand(K, B).unsqueeze(-1)
        target = logits_kbc.gather(dim=-1, index=y_exp).squeeze(-1)  # [K, B]

        mask = torch.ones_like(logits_kbc, dtype=torch.bool)
        mask.scatter_(dim=-1, index=y_exp, value=False)
        other = logits_kbc.masked_fill(~mask, torch.finfo(logits_kbc.dtype).min).max(dim=-1).values
        margin = target - other
        return F.softplus((float(self.loss_margin) - margin) / self._safe_tau()).mean(dim=0)

    def _prob_margin_loss(self, p_mean: torch.Tensor) -> torch.Tensor:
        """Hinge-like loss on posterior top-1 minus top-2 probability gap."""
        top2 = p_mean.topk(k=min(2, p_mean.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            gap = top2[..., 0]
        else:
            gap = top2[..., 0] - top2[..., 1]
        return F.softplus((float(self.loss_margin) - gap) / self._safe_tau())

    def _cos_margin_loss(self, feats: torch.Tensor, p_mean: torch.Tensor) -> torch.Tensor:
        """Last-layer cosine-margin loss using normalized features and classifier weights."""
        W = self.model.last_layer.weight
        feats_n = F.normalize(feats, dim=1)
        W_n = F.normalize(W, dim=1)
        cos_logits = F.linear(feats_n, W_n, None)
        y_hat = p_mean.detach().argmax(dim=-1)
        target = cos_logits.gather(dim=-1, index=y_hat.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(cos_logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=y_hat.unsqueeze(-1), value=False)
        other = cos_logits.masked_fill(~mask, torch.finfo(cos_logits.dtype).min).max(dim=-1).values
        margin = target - other
        return F.softplus((float(self.loss_margin) - margin) / self._safe_tau())

    def _energy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # Energy score E(x) = -logsumexp(logits). Minimizing it encourages confident low-energy predictions.
        return -torch.logsumexp(logits, dim=-1)

    def _batch_marginal_entropy(self, p_mean: torch.Tensor) -> torch.Tensor:
        # Entropy of the average prediction over the current batch; useful as an anti-collapse term.
        p_bar = p_mean.mean(dim=0).clamp_min(1e-12)
        p_bar = p_bar / p_bar.sum().clamp_min(1e-12)
        return entropy_from_probs(p_bar)

    def _as_betta_out(self, HE: torch.Tensor | dict, EH: torch.Tensor | None = None) -> dict:
        if isinstance(HE, dict):
            return HE
        betta = getattr(self, "_last_betta_out", None)
        if isinstance(betta, dict) and "p_mean" in betta:
            return betta
        return {"HE": HE, "EH": EH, "MI": HE - EH}

    def _betta_objective(self, HE: torch.Tensor | dict, EH: torch.Tensor | None = None) -> torch.Tensor:
        """Method-independent BeTTA loss over already-selected samples.

        Backward-compatible call patterns:
          - _betta_objective(HE, EH)
          - _betta_objective(betta_out)

        New losses use self._last_betta_out when the caller passes only HE/EH.
        """
        betta = self._as_betta_out(HE, EH)
        HE = betta["HE"]
        EH = betta["EH"]
        MI = betta.get("MI", HE - EH)
        sel = str(self.loss_type).lower().replace("-", "_")
        lam = float(getattr(self, "loss_lambda", 1.0))
        gamma = float(getattr(self, "loss_gamma", 1.0))

        # ------------------------------------------------------------------
        # Original HE/EH/MI families. Multipliers are kept for old grids.
        # ------------------------------------------------------------------
        if sel in {"a", "he"}:
            return HE.mean()
        if sel == "2a":
            return HE.mean() * 2.0
        if sel == "3a":
            return HE.mean() * 3.0
        if sel == "4a":
            return HE.mean() * 4.0

        if sel in {"b", "eh"}:
            return EH.mean()
        if sel == "2b":
            return EH.mean() * 2.0
        if sel == "3b":
            return EH.mean() * 3.0
        if sel == "4b":
            return EH.mean() * 4.0

        if sel in {"c", "he_eh", "he+eh"}:
            return HE.mean() + EH.mean()
        if sel == "2c":
            return (HE.mean() + EH.mean()) * 2.0
        if sel == "3c":
            return (HE.mean() + EH.mean()) * 3.0
        if sel == "4c":
            return (HE.mean() + EH.mean()) * 4.0

        if sel in {"d", "mi", "he_eh_minus", "he_minus_eh", "he-eh"}:
            return MI.mean()
        if sel == "2d":
            return MI.mean() * 2.0
        if sel == "3d":
            return MI.mean() * 3.0
        if sel == "4d":
            return MI.mean() * 4.0

        # From here on, the richer losses require the full betta_out dict.
        p_mean = betta.get("p_mean", None)
        probs_kbc = betta.get("probs_kbc", None)
        logits_kbc = betta.get("logits_kbc", None)
        base_logits = betta.get("base_logits", None)
        feats = betta.get("features", None)
        agreement = betta.get("agreement", None)
        mc_var = betta.get("mc_var", None)
        logit_var = betta.get("logit_var", None)

        # If an older caller did not provide rich outputs, degrade gracefully.
        if p_mean is None:
            if sel in {"mw_he", "mean_weighted_he", "miw_he", "mi_gate_he"}:
                w = self._low_score_weight(MI)
                return (w * HE).mean()
            raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires full betta_out from _betta_forward().")

        if mc_var is None and probs_kbc is not None:
            mc_var = probs_kbc.var(dim=0, unbiased=False).sum(dim=-1)
        if agreement is None:
            agreement = torch.ones_like(HE)

        # ------------------------------------------------------------------
        # 1) Mean-centered / reliability-weighted entropy variants.
        # ------------------------------------------------------------------
        if sel in {"mw_he", "mean_weighted_he", "miw_he", "centered_mi_he"}:
            w = self._low_score_weight(MI)
            return (w * HE).mean()

        if sel in {"mw_he_eh", "miw_he_eh"}:
            w = self._low_score_weight(MI)
            return (w * HE).mean() + gamma * EH.mean()

        if sel in {"mw_eh", "miw_eh", "mean_weighted_eh", "centered_mi_eh"}:
            # Use MI only as a reliability weight on E_H.
            # This tests whether sample-level MC entropy is the useful weighted target.
            w = self._low_score_weight(MI)
            return (w * EH).mean()

        if sel in {"he_mw_eh", "he_miw_eh", "he_plus_mw_eh", "he_plus_miw_eh"}:
            # Keep the main H_E sharpening term unweighted, but apply MI reliability
            # only to E_H. This is useful when raw H_E is a strong risk/adaptation
            # signal, while E_H needs reliability gating.
            w = self._low_score_weight(MI)
            return HE.mean() + gamma * (w * EH).mean()

        if sel in {"eh_mw_he", "eh_miw_he", "eh_plus_mw_he", "eh_plus_miw_he"}:
            # Symmetric diagnostic: keep E_H unweighted and gate H_E.
            w = self._low_score_weight(MI)
            return EH.mean() + gamma * (w * HE).mean()

        if sel in {"mw_he_plus_eh", "miw_he_plus_eh", "mw_he_unweighted_eh"}:
            # Alias for the existing mw_he_eh semantics: weighted H_E plus raw E_H.
            w = self._low_score_weight(MI)
            return (w * HE).mean() + gamma * EH.mean()

        mw_he_weh_scales = {
            "mw_he_weh": 1.0,
            "mw_he_weighted_eh": 1.0,
            "miw_he_weh": 1.0,
            "mw_both": 1.0,
            "miw_both": 1.0,
            "2mw_he_weh": 2.0,
            "3mw_he_weh": 3.0,
            "4mw_he_weh": 4.0,
        }
        if sel in mw_he_weh_scales:
            # Same low-MI reliability weight for both HE and EH.
            # This is usually safer than mw_he_eh when uncertain samples have large EH.
            w = self._low_score_weight(MI)
            return (w * (HE + gamma * EH)).mean() * mw_he_weh_scales[sel]

        if sel in {"mw_agree_he", "mi_agreement_weighted_he"}:
            # Low epistemic uncertainty AND high MC top-1 agreement.
            w_mi = self._low_score_weight(MI)
            w_ag = self._agreement_weight(agreement)
            w = self._normalize_loss_weights(w_mi * w_ag)
            return (w * HE).mean()

        if sel in {"mw_agree_he_eh", "mi_agreement_weighted_he_eh"}:
            # Weighted HE+EH using the same reliability weight.
            w_mi = self._low_score_weight(MI)
            w_ag = self._agreement_weight(agreement)
            w = self._normalize_loss_weights(w_mi * w_ag)
            return (w * (HE + gamma * EH)).mean()

        if sel in {"mw_var_he", "mi_var_weighted_he"}:
            # Low epistemic uncertainty AND low MC probability variance.
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            w_mi = self._low_score_weight(MI)
            w_var = self._low_score_weight(mc_var)
            w = self._normalize_loss_weights(w_mi * w_var)
            return (w * HE).mean()

        if sel in {"mw_var_he_eh", "mi_var_weighted_he_eh"}:
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            w_mi = self._low_score_weight(MI)
            w_var = self._low_score_weight(mc_var)
            w = self._normalize_loss_weights(w_mi * w_var)
            return (w * (HE + gamma * EH)).mean()

        if sel in {"mw_agree_var_he", "mi_agreement_var_weighted_he"}:
            # Stricter version: low-MI, high-agreement, low-MC-variance.
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            w_mi = self._low_score_weight(MI)
            w_ag = self._agreement_weight(agreement)
            w_var = self._low_score_weight(mc_var)
            w = self._normalize_loss_weights(w_mi * w_ag * w_var)
            return (w * HE).mean()

        if sel in {"mw_agree_var_he_eh", "mi_agreement_var_weighted_he_eh"}:
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            w_mi = self._low_score_weight(MI)
            w_ag = self._agreement_weight(agreement)
            w_var = self._low_score_weight(mc_var)
            w = self._normalize_loss_weights(w_mi * w_ag * w_var)
            return (w * (HE + gamma * EH)).mean()

        if sel in {"mw_conf_he", "mi_conf_weighted_he"}:
            # Low MI but also prefer confident posterior mean predictions.
            conf = p_mean.max(dim=-1).values
            w_mi = self._low_score_weight(MI)
            w_conf = self._normalize_loss_weights(conf.clamp_min(1e-12).pow(gamma))
            w = self._normalize_loss_weights(w_mi * w_conf)
            return (w * HE).mean()

        mw_conf_he_eh_scales = {
            "mw_conf_he_eh": 1.0,
            "mi_conf_weighted_he_eh": 1.0,
            "2mw_conf_he_eh": 2.0,
            "3mw_conf_he_eh": 3.0,
            "4mw_conf_he_eh": 4.0,
        }
        if sel in mw_conf_he_eh_scales:
            conf = p_mean.max(dim=-1).values
            w_mi = self._low_score_weight(MI)
            w_conf = self._normalize_loss_weights(conf.clamp_min(1e-12).pow(gamma))
            w = self._normalize_loss_weights(w_mi * w_conf)
            return (w * (HE + gamma * EH)).mean() * mw_conf_he_eh_scales[sel]

        if sel in {"mw_margin_he", "mi_margin_weighted_he"}:
            # Low MI and large posterior top-1/top-2 probability gap.
            top2 = p_mean.topk(k=min(2, p_mean.shape[-1]), dim=-1).values
            gap = top2[..., 0] if top2.shape[-1] == 1 else top2[..., 0] - top2[..., 1]
            w_mi = self._low_score_weight(MI)
            w_gap = self._normalize_loss_weights(torch.sigmoid((gap - float(self.loss_margin)) / self._safe_tau()))
            w = self._normalize_loss_weights(w_mi * w_gap)
            return (w * HE).mean()

        if sel in {"mw_margin_he_eh", "mi_margin_weighted_he_eh"}:
            top2 = p_mean.topk(k=min(2, p_mean.shape[-1]), dim=-1).values
            gap = top2[..., 0] if top2.shape[-1] == 1 else top2[..., 0] - top2[..., 1]
            w_mi = self._low_score_weight(MI)
            w_gap = self._normalize_loss_weights(torch.sigmoid((gap - float(self.loss_margin)) / self._safe_tau()))
            w = self._normalize_loss_weights(w_mi * w_gap)
            return (w * (HE + gamma * EH)).mean()

        if sel in {"mw_robust_he", "robust_mw_he"}:
            # Median/MAD-centered low-MI weighting; more robust to outlier batches than mean-centered mw_he.
            mi_det = MI.detach()
            med = mi_det.median()
            mad = (mi_det - med).abs().median().clamp_min(1e-6)
            z = ((MI - med) / (mad * self._safe_tau())).clamp(-20.0, 20.0)
            w = self._normalize_loss_weights(torch.exp(-z))
            return (w * HE).mean()

        if sel in {"mw_robust_he_eh", "robust_mw_he_eh"}:
            mi_det = MI.detach()
            med = mi_det.median()
            mad = (mi_det - med).abs().median().clamp_min(1e-6)
            z = ((MI - med) / (mad * self._safe_tau())).clamp(-20.0, 20.0)
            w = self._normalize_loss_weights(torch.exp(-z))
            return (w * (HE + gamma * EH)).mean()

        if sel in {"mw_safe_he", "safe_mw_he"}:
            # Avoid both extremely over-confident and nearly-uniform predictions.
            w_mi = self._low_score_weight(MI)
            he_norm = HE / math.log(max(int(self._num_classes_for_betta), 2))
            low_guard = torch.sigmoid((he_norm - 0.02) / self._safe_tau())
            high_guard = torch.sigmoid((0.70 - he_norm) / self._safe_tau())
            w = self._normalize_loss_weights(w_mi * low_guard * high_guard)
            return (w * HE).mean()

        if sel in {"mw_safe_he_eh", "safe_mw_he_eh"}:
            w_mi = self._low_score_weight(MI)
            he_norm = HE / math.log(max(int(self._num_classes_for_betta), 2))
            low_guard = torch.sigmoid((he_norm - 0.02) / self._safe_tau())
            high_guard = torch.sigmoid((0.70 - he_norm) / self._safe_tau())
            w = self._normalize_loss_weights(w_mi * low_guard * high_guard)
            return (w * (HE + gamma * EH)).mean()

        if sel in {"mw_im_eh", "weighted_im_eh"}:
            # Weighted sample entropy plus weighted EH, with batch-level anti-collapse.
            w = self._low_score_weight(MI)
            return (w * (HE + gamma * EH)).mean() - lam * self._batch_marginal_entropy(p_mean)

        if sel in {"mi_gate_he", "soft_gate_he"}:
            w = self._normalize_loss_weights(torch.sigmoid((MI.detach().mean() - MI) / self._safe_tau()))
            return (w * HE).mean()

        if sel in {"conf_he", "confidence_he"}:
            conf = p_mean.max(dim=-1).values
            w = self._normalize_loss_weights(conf.clamp_min(1e-12).pow(gamma))
            return (w * HE).mean()

        if sel in {"agree_he", "agreement_he"}:
            w = self._agreement_weight(agreement)
            return (w * HE).mean()

        if sel in {"var_weighted_he", "varw_he"}:
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            w = self._low_score_weight(mc_var)
            return (w * HE).mean()

        # ------------------------------------------------------------------
        # 2) MC posterior-mean consistency / variance losses.
        # ------------------------------------------------------------------
        if sel in {"var", "mc_var"}:
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            return mc_var.mean()

        if sel in {"he_var", "var_he"}:
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            return HE.mean() + lam * mc_var.mean()

        if sel in {"mw_he_var", "miw_he_var"}:
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            w = self._low_score_weight(MI)
            return (w * HE).mean() + lam * mc_var.mean()

        if sel in {"logit_var", "he_logit_var"}:
            if logit_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires logits_kbc.")
            if sel == "logit_var":
                return logit_var.mean()
            return HE.mean() + lam * logit_var.mean()

        if sel in {"kl_mean", "pmean_to_samples_kl", "mean_teacher_kl"}:
            if probs_kbc is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            return self._kl_pmean_to_samples(p_mean, probs_kbc).mean()

        if sel in {"kl_sample", "samples_to_pmean_kl", "sample_teacher_kl"}:
            if probs_kbc is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            return self._kl_samples_to_pmean(p_mean, probs_kbc).mean()

        if sel in {"sym_kl", "symmetric_kl"}:
            if probs_kbc is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            return (self._kl_pmean_to_samples(p_mean, probs_kbc) + self._kl_samples_to_pmean(p_mean, probs_kbc)).mean()

        if sel in {"he_kl", "he_cons", "he_mean_kl"}:
            if probs_kbc is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            return HE.mean() + lam * self._kl_pmean_to_samples(p_mean, probs_kbc).mean()

        if sel in {"he_sym_kl", "he_sym_cons"}:
            if probs_kbc is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            sym = self._kl_pmean_to_samples(p_mean, probs_kbc) + self._kl_samples_to_pmean(p_mean, probs_kbc)
            return HE.mean() + lam * sym.mean()

        if sel in {"he_js", "he_mi"}:
            return HE.mean() + lam * MI.mean()

        # ------------------------------------------------------------------
        # 3) Pseudo-label / sharpened posterior losses with MC reliability gates.
        # ------------------------------------------------------------------
        if sel in {"sharp_ce", "sharpen_ce", "posterior_sharpen"}:
            return self._sharpened_ce_from_pmean(p_mean).mean()

        if sel in {"agree_ce", "agreement_ce"}:
            w = self._agreement_weight(agreement)
            return (w * self._pseudo_ce_from_pmean(p_mean)).mean()

        if sel in {"mi_ce", "uncertainty_ce"}:
            w = self._low_score_weight(MI)
            return (w * self._pseudo_ce_from_pmean(p_mean)).mean()

        if sel in {"var_ce", "variance_ce"}:
            if mc_var is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires probs_kbc.")
            w = self._low_score_weight(mc_var)
            return (w * self._pseudo_ce_from_pmean(p_mean)).mean()

        if sel in {"agree_sharp_ce", "agreement_sharpen_ce"}:
            w = self._agreement_weight(agreement)
            return (w * self._sharpened_ce_from_pmean(p_mean)).mean()

        # ------------------------------------------------------------------
        # 4) Last-layer margin / geometry losses.
        # ------------------------------------------------------------------
        if sel in {"mc_margin", "robust_margin"}:
            if logits_kbc is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires logits_kbc.")
            return self._mc_margin_loss(logits_kbc, p_mean).mean()

        if sel in {"he_mc_margin", "he_robust_margin"}:
            if logits_kbc is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires logits_kbc.")
            return HE.mean() + lam * self._mc_margin_loss(logits_kbc, p_mean).mean()

        if sel in {"agree_margin", "agreement_margin"}:
            if logits_kbc is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires logits_kbc.")
            w = self._agreement_weight(agreement)
            return (w * self._mc_margin_loss(logits_kbc, p_mean)).mean()

        if sel in {"prob_margin", "p_margin"}:
            return self._prob_margin_loss(p_mean).mean()

        if sel in {"he_prob_margin", "he_p_margin"}:
            return HE.mean() + lam * self._prob_margin_loss(p_mean).mean()

        if sel in {"cos_margin", "lastlayer_cos_margin"}:
            if feats is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires features.")
            return self._cos_margin_loss(feats, p_mean).mean()

        if sel in {"he_cos_margin", "he_lastlayer_cos_margin"}:
            if feats is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires features.")
            return HE.mean() + lam * self._cos_margin_loss(feats, p_mean).mean()

        # ------------------------------------------------------------------
        # 5) Information-maximization / anti-collapse batch losses.
        # ------------------------------------------------------------------
        if sel in {"im", "information_maximization"}:
            return HE.mean() - lam * self._batch_marginal_entropy(p_mean)

        if sel in {"mw_im", "weighted_im"}:
            w = self._low_score_weight(MI)
            return (w * HE).mean() - lam * self._batch_marginal_entropy(p_mean)

        if sel in {"balance", "class_balance"}:
            return -self._batch_marginal_entropy(p_mean)

        # ------------------------------------------------------------------
        # 6) Energy-style losses. Use cautiously; these are intentionally optional.
        # ------------------------------------------------------------------
        if sel in {"energy", "free_energy"}:
            if base_logits is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires base_logits.")
            return self._energy_loss(base_logits).mean()

        if sel in {"he_energy", "energy_he"}:
            if base_logits is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires base_logits.")
            return HE.mean() + lam * self._energy_loss(base_logits).mean()

        if sel in {"mw_energy", "weighted_energy"}:
            if base_logits is None:
                raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} requires base_logits.")
            w = self._low_score_weight(MI)
            return (w * self._energy_loss(base_logits)).mean()

        raise ValueError(
            f"Unknown BETTA_LOSS_TYPE={self.loss_type}. "
            "Supported original: a/he, b/eh, c/he_eh, d/mi plus 2x/3x/4x variants. "
            "Supported optional: mw_he, mw_eh, mw_he_eh, he_mw_eh, eh_mw_he, "
            "mw_he_weh/mw_both plus 2mw/3mw/4mw variants, "
            "mw_agree_he, mw_agree_he_eh, mw_var_he, mw_var_he_eh, mw_agree_var_he, mw_agree_var_he_eh, "
            "mw_conf_he, mw_conf_he_eh plus 2mw/3mw/4mw variants, mw_margin_he, mw_margin_he_eh, "
            "mw_robust_he, mw_robust_he_eh, mw_safe_he, mw_safe_he_eh, mw_im_eh, "
            "mi_gate_he, conf_he, agree_he, var_weighted_he, var, he_var, mw_he_var, logit_var, he_logit_var, "
            "kl_mean, kl_sample, sym_kl, he_kl/he_cons, he_sym_kl, he_js, "
            "sharp_ce, agree_ce, mi_ce, var_ce, agree_sharp_ce, "
            "mc_margin, he_mc_margin, agree_margin, prob_margin, he_prob_margin, "
            "cos_margin, he_cos_margin, im, mw_im, balance, energy, he_energy, mw_energy."
        )

    def _betta_objective_terms(self, HE: torch.Tensor, EH: torch.Tensor, p_mean: torch.Tensor | None = None) -> torch.Tensor:
        """Elementwise BeTTA objective terms, shape [B].

        For filter-based methods that need per-sample terms, return the closest
        elementwise analogue. Full MC/geometry losses should call _betta_objective.
        """
        sel = str(self.loss_type).lower().replace("-", "_")
        MI = HE - EH
        if sel in {"a", "he", "2a", "3a", "4a"}:
            scale = 1.0
            if sel.startswith("2"):
                scale = 2.0
            elif sel.startswith("3"):
                scale = 3.0
            elif sel.startswith("4"):
                scale = 4.0
            return HE * scale
        if sel in {"b", "eh", "2b", "3b", "4b"}:
            scale = 1.0
            if sel.startswith("2"):
                scale = 2.0
            elif sel.startswith("3"):
                scale = 3.0
            elif sel.startswith("4"):
                scale = 4.0
            return EH * scale
        if sel in {"c", "he_eh", "he+eh", "2c", "3c", "4c"}:
            scale = 1.0
            if sel.startswith("2"):
                scale = 2.0
            elif sel.startswith("3"):
                scale = 3.0
            elif sel.startswith("4"):
                scale = 4.0
            return (HE + EH) * scale
        if sel in {"d", "mi", "he_eh_minus", "he_minus_eh", "he-eh", "2d", "3d", "4d"}:
            scale = 1.0
            if sel.startswith("2"):
                scale = 2.0
            elif sel.startswith("3"):
                scale = 3.0
            elif sel.startswith("4"):
                scale = 4.0
            return MI * scale
        if sel in {"mw_he", "mean_weighted_he", "miw_he", "centered_mi_he", "mi_gate_he", "soft_gate_he"}:
            return self._low_score_weight(MI) * HE
        if sel in {"mw_he_eh", "miw_he_eh", "mw_he_plus_eh", "miw_he_plus_eh", "mw_he_unweighted_eh"}:
            return self._low_score_weight(MI) * HE + float(getattr(self, "loss_gamma", 1.0)) * EH
        if sel in {"mw_eh", "miw_eh", "mean_weighted_eh", "centered_mi_eh"}:
            return self._low_score_weight(MI) * EH
        if sel in {"he_mw_eh", "he_miw_eh", "he_plus_mw_eh", "he_plus_miw_eh"}:
            return HE + float(getattr(self, "loss_gamma", 1.0)) * self._low_score_weight(MI) * EH
        if sel in {"eh_mw_he", "eh_miw_he", "eh_plus_mw_he", "eh_plus_miw_he"}:
            return EH + float(getattr(self, "loss_gamma", 1.0)) * self._low_score_weight(MI) * HE
        mw_he_weh_scales = {
            "mw_he_weh": 1.0,
            "mw_he_weighted_eh": 1.0,
            "miw_he_weh": 1.0,
            "mw_both": 1.0,
            "miw_both": 1.0,
            "2mw_he_weh": 2.0,
            "3mw_he_weh": 3.0,
            "4mw_he_weh": 4.0,
        }
        if sel in mw_he_weh_scales:
            terms = self._low_score_weight(MI) * (HE + float(getattr(self, "loss_gamma", 1.0)) * EH)
            return terms * mw_he_weh_scales[sel]

        # Confidence-weighted variants require posterior predictive probabilities.
        # Filter-based methods such as DeYO/EATA/SAR call this elementwise helper on
        # selected samples, so p_mean must be sliced with the same selected indices
        # and passed in by the caller.
        if sel in {"mw_conf_he", "mi_conf_weighted_he"}:
            if p_mean is None:
                raise ValueError(
                    f"BETTA_LOSS_TYPE={self.loss_type} requires p_mean for elementwise confidence weighting."
                )
            gamma = float(getattr(self, "loss_gamma", 1.0))
            conf = p_mean.max(dim=-1).values
            w_mi = self._low_score_weight(MI)
            w_conf = self._normalize_loss_weights(conf.clamp_min(1e-12).pow(gamma))
            w = self._normalize_loss_weights(w_mi * w_conf)
            return w * HE

        mw_conf_he_eh_scales = {
            "mw_conf_he_eh": 1.0,
            "mi_conf_weighted_he_eh": 1.0,
            "2mw_conf_he_eh": 2.0,
            "3mw_conf_he_eh": 3.0,
            "4mw_conf_he_eh": 4.0,
        }
        if sel in mw_conf_he_eh_scales:
            if p_mean is None:
                raise ValueError(
                    f"BETTA_LOSS_TYPE={self.loss_type} requires p_mean for elementwise confidence weighting."
                )
            gamma = float(getattr(self, "loss_gamma", 1.0))
            conf = p_mean.max(dim=-1).values
            w_mi = self._low_score_weight(MI)
            w_conf = self._normalize_loss_weights(conf.clamp_min(1e-12).pow(gamma))
            w = self._normalize_loss_weights(w_mi * w_conf)
            return w * (HE + gamma * EH) * mw_conf_he_eh_scales[sel]

        raise ValueError(f"BETTA_LOSS_TYPE={self.loss_type} has no simple elementwise term.")

    @torch.no_grad()
    def _record_betta_trace(self, betta_out: dict):
        sigma = betta_out["sigma"]
        stats = betta_out["stats"]
        HE = betta_out["HE"]
        EH = betta_out["EH"]
        p = betta_out["p_mean"]

        self.sigma_log.append(float(sigma))
        item = {
            "selected_sigma": float(sigma),
            "selected_idx": int(stats.get("idx", -1)),
            "selected_idx_anchor": int(stats.get("idx_anchor", stats.get("idx", -1))),
            "selected_idx_nearest": int(stats.get("idx_nearest", stats.get("idx", -1))),
            "selected_sigma_grid_value": float(stats.get("sigma_grid_value", float("nan"))),
            "selected_sigma_grid_value_anchor": float(stats.get("sigma_grid_value_anchor", stats.get("sigma_grid_value", float("nan")))),
            "selected_sigma_grid_value_nearest": float(stats.get("sigma_grid_value_nearest", float("nan"))),
            "selected_sigma_fractional_idx": float(stats.get("sigma_fractional_idx", float("nan"))),
            "sigma_interpolated": bool(stats.get("sigma_interpolated", False)),
            "sigma_interp_kind": str(stats.get("sigma_interp_kind", "")),
            "selected_at_boundary": bool(stats.get("boundary", False)),
            "sigma_selection": self.selection,
            "sigma_scale_mode": self.model.sigma_scale_mode,
            "bias_scale_mode": self.model.bias_scale_mode,
            "HE_mean": float(HE.mean()),
            "EH_mean": float(EH.mean()),
            "MI_mean": float((HE - EH).mean()),
            "conf_mean": float(p.max(dim=1).values.mean()),
            "agreement_mean": float(betta_out.get("agreement", torch.ones_like(HE)).mean()),
            "mc_var_mean": float(betta_out.get("mc_var", torch.zeros_like(HE)).mean()),
            "logit_var_mean": float(betta_out.get("logit_var", torch.zeros_like(HE)).mean()),
            "posterior_scale_mode": str(getattr(self.model, "posterior_scale_mode", "scalar")),
            "aniso_strength": float(getattr(self.model, "aniso_strength", float("nan"))),
            "aniso_rank": int(getattr(self.model, "aniso_rank", 1)),
            "factor_grid_enabled": bool(stats.get("factor_grid_enabled", False)),
            "factor_combo_idx": int(stats.get("factor_combo_idx", -1)),
            "factor_selected_strength": float(stats.get("factor_selected_strength", getattr(self.model, "aniso_strength", float("nan")))),
            "factor_selected_rank": int(stats.get("factor_selected_rank", getattr(self.model, "aniso_rank", 1))),
            "grid_sigmas": self.sigma_grid.detach().cpu(),
        }
        item.update(self.model.posterior_std_summary())
        for key in ("EH", "HE", "MI", "joint", "HE_norm", "agreement", "dEH", "dHE", "dMI"):
            if key in stats:
                item[f"grid_{key}"] = stats[key]
        for key in (
            "factor_strength_grid", "factor_rank_grid", "factor_combo_strength", "factor_combo_rank",
            "factor_grid_shape", "factor_grid_EH", "factor_grid_HE", "factor_grid_MI",
            "factor_grid_joint", "factor_grid_HE_norm", "factor_grid_agreement",
            "factor_grid_dEH", "factor_grid_dHE", "factor_grid_dMI",
        ):
            if key in stats:
                item[key] = stats[key]
        self.scale_trace.append(item)


# =============================================================================
# 4. Tent-style norm adaptation utilities
# =============================================================================


class TentNormAdaptationMixin:
    """Norm-only adaptation, separated from BeTTA prediction logic."""

    def collect_params(self):
        params, names = [], []
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)

        for module_name, module in self.model.named_modules():
            if isinstance(module, norm_types):
                for param_name, param in module.named_parameters(recurse=False):
                    if param_name in {"weight", "bias"} and param.requires_grad:
                        params.append(param)
                        names.append(f"{module_name}.{param_name}")
        return params, names

    def configure_model(self):
        """Eval globally, update only norm affine params.

        BatchNorm2d/3d uses batch statistics without enabling dropout/stochastic-depth.
        """
        self.model.eval()
        self.model.requires_grad_(False)

        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.eval()
            elif isinstance(module, nn.BatchNorm1d):
                module.requires_grad_(True)
                if self.force_bn1d_batch_stats:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
                module.eval()
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                module.requires_grad_(True)
                module.eval()
            elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.eval()
