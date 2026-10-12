"""Small public adapter around the byte-preserved scientific MIRA core."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import os
import threading

import torch
from torch import nn
from ._reference_core import BeTTACoreMixin, TentNormAdaptationMixin, find_last_linear

SIGMA_GRID = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2,
              1e-1, 3e-1, 5e-1, 1., 2., 3., 4., 5.)
_INIT_LOCK = threading.RLock()


@dataclass(frozen=True)
class MIRAConfig:
    alpha: float = .03
    rho: float = .90
    tau: float = .25
    gamma: float = 1.
    K: int = 300
    sigma_grid: tuple = SIGMA_GRID
    scale_parameterization: str = "feature"
    base_scale: str = "he_in"
    loss: str = "full"
    lr: float = 1e-3
    optimizer: str = "SGD"
    momentum: float = .9
    nesterov: bool = True
    steps: int = 1
    gradient_norm_matching: bool = False
    gradient_norm_target: float = 1.

    def __post_init__(self):
        if not 0 <= self.alpha <= 1 or not 0 <= self.rho <= 1:
            raise ValueError("alpha and rho must lie in [0,1]")
        if self.tau <= 0 or self.K < 1 or self.steps < 1 or self.gamma < 0:
            raise ValueError("Invalid temperature, sampling budget, steps or EH coefficient")
        if len(self.sigma_grid) < 2 or any(s <= 0 for s in self.sigma_grid):
            raise ValueError("The scale grid must contain at least two positive radii")
        if tuple(sorted(set(self.sigma_grid))) != tuple(self.sigma_grid):
            raise ValueError("Scale candidates must be distinct and increasing")
        if self.loss not in {"full","he","eh","mi","he_eh","mi_he","mi_gate","confidence_only"}:
            raise ValueError("Unsupported objective")
        if self.optimizer not in {"SGD","Adam"}:
            raise ValueError("Supported optimizers: SGD, Adam")


@contextmanager
def _reference_settings(config):
    """Initialize fixed instance settings without inheriting research-shell flags."""
    loss = dict(full="mw_conf_he_eh", he="he", eh="eh", mi="mi", he_eh="he_eh",
                mi_he="mw_he", mi_gate="mw_he_weh", confidence_only="mw_conf_he_eh")[config.loss]
    values = {
        "BETTA_NUM_MC": config.K, "BETTA_LOSS_TYPE": loss,
        "BETTA_SIGMA_SELECTION": "mi_elbow_agreement", "BETTA_SAME_NOISE": 1,
        "BETTA_SIGMA_INTERPOLATE": 1, "BETTA_RETURN_LOG_PROBS": 1,
        "BETTA_RETURN_AFTER_ADAPT": 0, "BETTA_FACTOR_GRID": 0,
        "BETTA_SIGMA_SCALE_MODE": config.base_scale, "BETTA_BIAS_SCALE_MODE": "auto",
        "BETTA_POSTERIOR_SCALE_MODE": config.scale_parameterization,
        "BETTA_ANISO_STRENGTH": 1, "BETTA_ANISO_MIN": .25, "BETTA_ANISO_MAX": 4.,
        "BETTA_FORCE_BN1D_BATCH_STATS": 0, "BETTA_MI_ELBOW_ALPHA": config.alpha,
        "BETTA_AGREEMENT_THRESHOLD": config.rho, "BETTA_LOSS_TAU": config.tau,
        "BETTA_LOSS_LAMBDA": 1, "BETTA_LOSS_GAMMA": config.gamma,
        "BETTA_LOSS_DETACH_WEIGHTS": 1,
        "BETTA_GRAD_NORM_MATCH": int(config.gradient_norm_matching),
        "BETTA_GRAD_NORM_TARGET": config.gradient_norm_target,
        "BETTA_SIGMA_GRID": ",".join(map(str, config.sigma_grid)),
    }
    with _INIT_LOCK:
        saved = {k:v for k,v in os.environ.items() if k.startswith(("BETTA_","TTA_GRAD_"))}
        try:
            for k in saved: os.environ.pop(k)
            os.environ.update({k:str(v) for k,v in values.items()})
            yield
        finally:
            for k in list(os.environ):
                if k.startswith(("BETTA_","TTA_GRAD_")): os.environ.pop(k)
            os.environ.update(saved)


class MIRA(BeTTACoreMixin, TentNormAdaptationMixin, nn.Module):
    """Norm-only adaptation with pre-update log-probabilities returned per batch.

    The supplied model must already contain its source weights and input normalization.
    Call reset() at corruption boundaries for reset-based evaluation. Labels are never
    accepted by this adapter. K draws are shared across radii; fresh draws supply loss.
    """
    def __init__(self, model, config=None):
        nn.Module.__init__(self)
        self.config = config or MIRAConfig()
        with _reference_settings(self.config):
            self._init_betta_core(num_classes=find_last_linear(model).out_features)
            self.model = self._wrap_model_for_betta(model)
            self.configure_model()
            self.params, self.parameter_names = self.collect_params()
            if not self.params:
                raise ValueError("Model has no trainable normalization affine parameters")
            if self.config.optimizer == "SGD":
                self.optimizer = torch.optim.SGD(self.params, lr=self.config.lr,
                    momentum=self.config.momentum, nesterov=self.config.nesterov, weight_decay=0.)
            else:
                self.optimizer = torch.optim.Adam(self.params, lr=self.config.lr,
                    betas=(.9,.999), weight_decay=0.)
            self._init_sigma_grid_after_super()
        self._source_state = deepcopy(self.model.state_dict())
        self._source_optimizer = deepcopy(self.optimizer.state_dict())
        self.last_diagnostics = None
        self.last_weights = None

    def _match_gradient_norm(self, loss: torch.Tensor) -> torch.Tensor:
        """Rescale a scalar loss to a fixed adapted-parameter gradient L2 norm."""
        if not self.grad_norm_match:
            return loss
        if loss.ndim != 0:
            raise ValueError("Gradient-norm matching requires a scalar loss.")

        params = [p for p in self.params if p.requires_grad]
        if not params or not loss.requires_grad:
            self.last_raw_grad_norm = 0.0
            self.last_grad_norm_scale = 1.0
            return loss

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        sq_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
        for grad in grads:
            if grad is not None:
                sq_norm = sq_norm + grad.detach().float().square().sum()
        raw_norm = sq_norm.sqrt()

        eps = max(self.grad_norm_eps, 0.0)
        target = max(self.grad_norm_target, 0.0)
        max_scale = max(self.grad_norm_max_scale, 0.0)
        if (not bool(torch.isfinite(raw_norm).item())) or float(raw_norm.item()) <= eps:
            scale = raw_norm.new_tensor(1.0)
        else:
            scale = raw_norm.new_tensor(target) / raw_norm.clamp_min(eps)
            if max_scale > 0.0:
                scale = scale.clamp(max=max_scale)

        self.last_raw_grad_norm = float(raw_norm.item())
        self.last_grad_norm_scale = float(scale.item())
        return loss * scale.to(device=loss.device, dtype=loss.dtype).detach()

    def _objective(self, betta):
        H,E = betta["HE"],betta["EH"]
        if self.config.loss in {"full","confidence_only"}:
            confidence = betta["p_mean"].max(-1).values
            wc = self._normalize_loss_weights(confidence.clamp_min(1e-12))
            w = wc if self.config.loss == "confidence_only" else self._normalize_loss_weights(
                self._low_score_weight(H-E)*wc)
            self.last_weights = w
            # gamma is exclusively the EH coefficient; confidence exponent stays 1.
            return (w*(H+self.config.gamma*E)).mean()
        self.last_weights = None
        return self._betta_objective(H,E)

    @torch.enable_grad()
    def forward(self, images):
        for _ in range(self.config.steps):
            betta = self._betta_forward(images)
            output = betta["output"]
            loss = self._match_betta_gradient_norm(self._objective(betta))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._record_betta_trace(betta)
            self.last_diagnostics = self.scale_trace[-1]
            self.scale_trace.clear()
            self.sigma_log.clear()
        return output

    def predict_proba(self, images):
        return self(images).softmax(-1)

    def reset(self):
        self.model.load_state_dict(self._source_state, strict=True)
        self.optimizer.load_state_dict(deepcopy(self._source_optimizer))
        self.optimizer.zero_grad()
        self.scale_trace.clear()
        self.sigma_log.clear()
        self.last_diagnostics = self.last_weights = None
