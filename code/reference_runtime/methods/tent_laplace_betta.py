"""Tent + BeTTA. Shared BeTTA code lives in methods/betta_core.py."""

import torch

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from methods.betta_core import BeTTACoreMixin, TentNormAdaptationMixin


@ADAPTATION_REGISTRY.register()
class Tent_laplace_betta(BeTTACoreMixin, TentNormAdaptationMixin, TTAMethod):
    """BeTTA-style MC last-layer prediction + Tent-style norm adaptation."""

    def __init__(self, cfg, model, num_classes):
        self._init_betta_core(num_classes=num_classes)
        wrapped_model = self._wrap_model_for_betta(model)
        super().__init__(cfg, wrapped_model, num_classes)
        self._init_sigma_grid_after_super()

    def loss_calculation(self, x):
        imgs_test = x[0]
        betta = self._betta_forward(imgs_test)
        outputs = betta["output"]
        loss = self._betta_objective(betta["HE"], betta["EH"])

        self._record_betta_trace(betta)
        if self.check_update:
            self._record_update_trace(outputs, perform_update=True)
        return outputs, loss

    @torch.no_grad()
    def _predict_after_update(self, x):
        return self._betta_forward(x[0])["output"]

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            loss = self._match_betta_gradient_norm(loss)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs, loss = self.loss_calculation(x)
            loss = self._match_betta_gradient_norm(loss)
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        final_outputs = self._predict_after_update(x) if self.return_after_adapt else outputs
        self._record_prediction_reliability(final_outputs)
        return final_outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.sigma_log = []
        self.scale_trace = []
