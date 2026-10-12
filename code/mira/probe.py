"""Detached reliability weights for explicit, method-specific integration."""
import torch
from .adapter import MIRAConfig, _reference_settings
from ._reference_core import BeTTACoreMixin, find_last_linear


class ReliabilityProbe(BeTTACoreMixin):
    """Probe an already configured base model without changing its trainable set.

    This helper performs no optimizer update. In a SAM integration, call it at the
    unperturbed state once, then apply the returned fixed weights at both steps.
    Applying weights after the base method's sample filtering avoids changing its
    filtering criterion. This interface is not a claim to reproduce retained hybrid
    EATA/SAR/ROID table values.
    """
    def __init__(self,model,config=None):
        self.config=config or MIRAConfig()
        with _reference_settings(self.config):
            self._init_betta_core(num_classes=find_last_linear(model).out_features)
            self.model=self._wrap_model_for_betta(model)
            self._init_sigma_grid_after_super()

    @torch.no_grad()
    def weights(self,images):
        betta=self._betta_forward(images)
        wc=self._normalize_loss_weights(betta['p_mean'].max(-1).values.clamp_min(1e-12))
        w=self._normalize_loss_weights(self._low_score_weight(betta['HE']-betta['EH'])*wc)
        self._record_betta_trace(betta)
        diagnostics=self.scale_trace[-1]
        self.scale_trace.clear();self.sigma_log.clear()
        return w,diagnostics
