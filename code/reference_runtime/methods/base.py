import logging
import os
import torch
import torch.nn as nn
from torchvision import transforms

from copy import deepcopy
from functools import wraps
import math

import torch.nn.functional as F



logger = logging.getLogger(__name__)


class TTAMethod(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.episodic = cfg.MODEL.EPISODIC
        self.dataset_name = cfg.CORRUPTION.DATASET
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables for resetting the model after a certain amount of performed update steps
        self.performed_updates = 0
        self.reset_after_num_updates = cfg.MODEL.RESET_AFTER_NUM_UPDATES

        # restore the image input size from the model pre-processing if it is defined
        # this is required for methods relying on test-time augmentation
        self.img_size = (32, 32) if "cifar" in self.dataset_name else (224, 224)
        if hasattr(self.model, "model_preprocess") and isinstance(self.model.model_preprocess, transforms.Compose):
            for transf in self.model.model_preprocess.transforms[::-1]:
                if hasattr(transf, "size"):
                    self.img_size = getattr(transf, "size")
                    if self.dataset_name in ["imagenet_c", "ccc"] and max(self.img_size) > 224:
                        raise ValueError(f"The specified model with pre-processing {model.model_preprocess} "
                                         f"is not suited in combination with ImageNet-C and CCC! "
                                         f"These datasets are already resized and center cropped to 224")
                    break

        # configure model and optimizer
        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.num_trainable_params, self.num_total_params = self.get_number_trainable_params()

        # Optional gradient-strength matching shared by plain TTA methods and
        # BeTTA variants. TTA_* names take precedence; BETTA_* names remain as
        # backward-compatible aliases used by the experiment launchers.
        self.grad_norm_match = bool(int(os.environ.get(
            "TTA_GRAD_NORM_MATCH",
            os.environ.get("BETTA_GRAD_NORM_MATCH", "0"),
        )))
        self.grad_norm_target = float(os.environ.get(
            "TTA_GRAD_NORM_TARGET",
            os.environ.get("BETTA_GRAD_NORM_TARGET", "1.0"),
        ))
        self.grad_norm_eps = float(os.environ.get(
            "TTA_GRAD_NORM_EPS",
            os.environ.get("BETTA_GRAD_NORM_EPS", "1e-12"),
        ))
        self.grad_norm_max_scale = float(os.environ.get(
            "TTA_GRAD_NORM_MAX_SCALE",
            os.environ.get("BETTA_GRAD_NORM_MAX_SCALE", "1000000.0"),
        ))
        self.last_raw_grad_norm = float("nan")
        self.last_grad_norm_scale = 1.0


        self.landscape_enabled = bool(getattr(cfg.TEST, "LANDSCAPE", False))
        self.landscape_stride = max(1, int(getattr(cfg.TEST, "LANDSCAPE_STRIDE", 1)))
        self.landscape_max_points = int(getattr(cfg.TEST, "LANDSCAPE_MAX_POINTS", 200))
        self.landscape_param_mode = str(getattr(cfg.TEST, "LANDSCAPE_PARAM_MODE", "all_norm"))

        self.landscape_params, self.landscape_param_names = self.collect_landscape_params(
            mode=self.landscape_param_mode
        )

        self.landscape_trajectory = []
        self.landscape_step = 0




        # variables needed for single sample test-time adaptation (sstta) using a sliding window (buffer) approach
        self.input_buffer = None
        self.window_length = cfg.TEST.WINDOW_LENGTH
        self.pointer = torch.tensor([0], dtype=torch.long).to(self.device)
        # sstta: if the model has no batchnorm layers, we do not need to forward the whole buffer when not performing any updates
        self.has_bn = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # setup for mixed-precision or single precision
        self.mixed_precision = cfg.MIXED_PRECISION
        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None

        # Optional diagnosis: record which samples actually contribute to TTA updates.
        # Disabled by default to keep normal runs clean.
        self.check_update = bool(getattr(cfg.TEST, "CHECK_UPDATE", False))
        self._current_update_trace_steps = []
        self.last_update_trace = {"steps": []}

        self.adaptation_enabled = True

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


    def collect_landscape_params(self, mode="all_norm"):
        """
        For multi-method comparison, use the same parameter vector space.

        mode:
        - "all_norm": all BN/LN/GN affine params, even if a method does not update them
        - "adapted": only this method's adapted params
        """
        if mode == "adapted":
            return list(self.params), list(getattr(self, "param_names", []))

        if mode != "all_norm":
            raise ValueError(f"Unknown LANDSCAPE_PARAM_MODE={mode}")

        params = []
        names = []

        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names


    @torch.no_grad()
    def _flat_landscape_params(self):
        return torch.cat([
            p.detach().float().cpu().reshape(-1)
            for p in self.landscape_params
        ])


    @torch.no_grad()
    def _assign_landscape_params(self, vec):
        offset = 0
        for p in self.landscape_params:
            n = p.numel()
            chunk = vec[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype)
            p.copy_(chunk)
            offset += n


    @torch.no_grad()
    def start_landscape_trace(self):
        self.landscape_trajectory = [self._flat_landscape_params()]
        self.landscape_step = 0


    @torch.no_grad()
    def _record_landscape_step(self):
        if not self.landscape_enabled:
            return

        self.landscape_step += 1

        if self.landscape_step % self.landscape_stride != 0:
            return

        if self.landscape_max_points > 0 and len(self.landscape_trajectory) >= self.landscape_max_points:
            return

        self.landscape_trajectory.append(self._flat_landscape_params())


    @torch.no_grad()
    def get_landscape_trajectory(self):
        if len(self.landscape_trajectory) == 0:
            return None
        return torch.stack(self.landscape_trajectory, dim=0)


    def get_landscape_param_names(self):
        return list(self.landscape_param_names)




    def set_adaptation_enabled(self, enabled: bool):
        self.adaptation_enabled = bool(enabled)

    def _init_update_trace(self):
        if not self.check_update:
            return
        self._current_update_trace_steps = []
        self.last_update_trace = {"steps": []}

    @torch.no_grad()
    def _record_update_trace(self, outputs, selected_indices=None, selected_mask=None, perform_update=True, extra=None):
        """Record which samples were used to compute the adaptation loss.

        The evaluator can later compare `selected_predictions` with labels.
        Labels are deliberately not used here, so adaptation remains label-free.
        """
        if (not self.check_update) or outputs is None:
            return

        batch_size = int(outputs.shape[0])
        device = outputs.device

        if selected_indices is None:
            if selected_mask is None:
                if perform_update:
                    selected_indices = torch.arange(batch_size, device=device, dtype=torch.long)
                else:
                    selected_indices = torch.empty(0, device=device, dtype=torch.long)
            else:
                selected_mask = selected_mask.to(device)
                if selected_mask.dtype == torch.bool:
                    selected_indices = torch.where(selected_mask)[0]
                else:
                    selected_indices = selected_mask.long()
        else:
            selected_indices = selected_indices.to(device=device, dtype=torch.long)

        if not perform_update:
            selected_indices = selected_indices[:0]

        logits = outputs.detach()
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        step_trace = {
            "batch_size": batch_size,
            "perform_update": bool(perform_update and selected_indices.numel() > 0),
            "selected_indices": selected_indices.detach().cpu(),
            "selected_predictions": preds[selected_indices].detach().cpu(),
            "selected_confidences": confs[selected_indices].detach().cpu(),
            "all_predictions": preds.detach().cpu(),
            "all_confidences": confs.detach().cpu(),
        }

        if extra:
            safe_extra = {}
            for key, value in extra.items():
                if torch.is_tensor(value):
                    safe_extra[key] = value.detach().cpu()
                else:
                    safe_extra[key] = value
            step_trace["extra"] = safe_extra

        self._current_update_trace_steps.append(step_trace)



    def _entropy_from_probs_for_reliability(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.float().clamp_min(1e-12)
        return -(probs * probs.log()).sum(dim=-1)


    def _margin_from_probs_for_reliability(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.float()
        if probs.size(-1) < 2:
            return probs.max(dim=-1).values
        top2 = probs.topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1]


    @torch.no_grad()
    def _record_prediction_reliability(self, outputs: torch.Tensor):
        """
        Minimal label-free prediction metrics for diagnostic plots.

        No labels, correctness, or batch accuracy are used here.
        These values are only computed from returned logits/probabilities.
        """
        if outputs is None:
            return {}

        probs = F.softmax(outputs.detach().float(), dim=1).clamp_min(1e-12)
        num_classes = max(int(probs.size(1)), 2)
        max_entropy = math.log(float(num_classes))

        confidence = probs.max(dim=1).values
        margin = self._margin_from_probs_for_reliability(probs)
        entropy = self._entropy_from_probs_for_reliability(probs)
        entropy_norm = entropy / max(max_entropy, 1e-12)

        base_confidence = float(confidence.mean().item())
        confidence_margin = float(margin.mean().item())
        base_entropy_norm = float(entropy_norm.mean().item())

        # Simple label-free risk proxy.  This is not supervised calibration.
        risk_terms = [
            1.0 - base_confidence,
            1.0 - confidence_margin,
            base_entropy_norm,
        ]
        label_free_risk = float(sum(risk_terms) / len(risk_terms))
        label_free_risk = max(0.0, min(1.0, label_free_risk))
        label_free_reliability = 1.0 - label_free_risk

        metrics = {
            "base_confidence": base_confidence,
            "confidence_margin": confidence_margin,
            "base_entropy_norm": base_entropy_norm,
            "label_free_risk": label_free_risk,
            "label_free_reliability": label_free_reliability,
        }

        if not hasattr(self, "last_uncertainty") or self.last_uncertainty is None:
            self.last_uncertainty = {}

        self.last_uncertainty.update(metrics)
        return metrics


    def trajectory_uncertainty(self):
        """
        Compatibility hook for online trace collection.
        """
        return deepcopy(getattr(self, "last_uncertainty", {}))













    def forward(self, x):
        if self.episodic:
            self.reset()

        x = x if isinstance(x, list) else [x]
        self._init_update_trace()

        # Evaluation-only forward: no TTA update.
        # Used for partial-adaptation experiments after the warmup/cutoff batches.
        if not getattr(self, "adaptation_enabled", True):
            with torch.no_grad():
                outputs = self.forward_sliding_window(x)

            if self.check_update:
                self._record_update_trace(outputs, perform_update=False)
                self.last_update_trace = {"steps": list(self._current_update_trace_steps)}

            return outputs





        if x[0].shape[0] == 1:  # single sample test-time adaptation
            # create the sliding window input
            if self.input_buffer is None:
                self.input_buffer = [x_item for x_item in x]
                # set bn1d layers into eval mode, since no statistics can be extracted from 1 sample
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=False)
            elif self.input_buffer[0].shape[0] < self.window_length:
                self.input_buffer = [torch.cat([self.input_buffer[i], x_item], dim=0) for i, x_item in enumerate(x)]
                # set bn1d layers into train mode
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=True)
            else:
                for i, x_item in enumerate(x):
                    self.input_buffer[i][self.pointer] = x_item

            if self.pointer == (self.window_length - 1):
                # update the model, since the complete buffer has changed
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(self.input_buffer)

                    self._record_landscape_step()

                    self.performed_updates += 1
                    if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                        self.reset()

                outputs = outputs[self.pointer.long()]
            else:
                # create the prediction without updating the model
                if self.has_bn:
                    # forward the whole buffer to get good batchnorm statistics
                    outputs = self.forward_sliding_window(self.input_buffer)
                    outputs = outputs[self.pointer.long()]
                else:
                    # only forward the current test sample, since there are no batchnorm layers
                    outputs = self.forward_sliding_window(x)

            # increase the pointer
            self.pointer += 1
            self.pointer %= self.window_length

        else:   # common batch adaptation setting
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)


                self._record_landscape_step()

                # if specified, reset the model after a certain amount of update steps
                self.performed_updates += 1
                if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                    self.reset()

        if self.check_update:
            self.last_update_trace = {"steps": list(self._current_update_trace_steps)}
        return outputs

    def loss_calculation(self, x):
        """
        Loss calculation.
        """
        raise NotImplementedError

    def forward_and_adapt(self, x):
        """
        Forward and adapt the model on a batch of data.
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model(imgs_test)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        if self.cfg.OPTIM.METHOD == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=self.cfg.OPTIM.LR,
                                    betas=(self.cfg.OPTIM.BETA, 0.999),
                                    weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'AdamW':
            return torch.optim.AdamW(self.params,
                                     lr=self.cfg.OPTIM.LR,
                                     betas=(self.cfg.OPTIM.BETA, 0.999),
                                     weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'SGD':
            return torch.optim.SGD(self.params,
                                   lr=self.cfg.OPTIM.LR,
                                   momentum=self.cfg.OPTIM.MOMENTUM,
                                   dampening=self.cfg.OPTIM.DAMPENING,
                                   weight_decay=self.cfg.OPTIM.WD,
                                   nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    def get_number_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"#Trainable/total parameters: {trainable:,}/{total:,} \t Ratio: {trainable / total * 100:.3f}% ")
        return trainable, total

    def reset(self):
        """Reset the model and optimizer state to the initial source state"""
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def copy_model(model):
        return deepcopy(model)

    @staticmethod
    def change_mode_of_batchnorm1d(model_list, to_train_mode=True):
        # batchnorm1d layers do not work with single sample inputs
        for model in model_list:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    if to_train_mode:
                        m.train()
                    else:
                        m.eval()


def forward_decorator(fn):
    @wraps(fn)
    def decorator(self, *args, **kwargs): 
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = fn(self, *args, **kwargs)
        else:
            outputs = fn(self, *args, **kwargs)
        return outputs
    return decorator
