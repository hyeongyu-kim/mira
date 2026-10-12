import torch
import torch.nn as nn
import math

class Entropy(nn.Module):
    def __init__(self, temperature=1.0, weight_by_temperature=True, normalize=False):
        super().__init__()
        self.temperature = float(temperature)
        self.weight_by_temperature = bool(weight_by_temperature)
        self.normalize = bool(normalize)
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def __call__(self, logits):
        scaled_logits = logits / self.temperature
        probs = scaled_logits.softmax(-1)
        log_probs = scaled_logits.log_softmax(-1)
        entropy = -(probs * log_probs).sum(-1)

        if self.weight_by_temperature:
            entropy = entropy * self.temperature

        if self.normalize:
            entropy = entropy / math.log(logits.shape[-1])

        return entropy




# class NosoftEntropy(nn.Module):
#     def __init__(self):
#         super(NosoftEntropy, self).__init__()

#     def __call__(self, probs): # (변수명을 logits에서 probs로 변경)
#         # 0*log(0) = NaN이 되는 것을 막기 위해 eps clamp
#         probs_clamped = probs.clamp(min=1e-20)
        
#         # ★★★ 핵심 수정 ★★★
#         # sum(1) -> sum(-1)
#         # 1번 차원이 아닌, 항상 마지막 차원(Class)을 합산합니다.
#         return -(probs_clamped * probs_clamped.log()).sum(-1)

class NosoftEntropy(nn.Module):
    def __init__(self, temperature=1.0, weight_by_temperature=True, eps=1e-20):
        super(NosoftEntropy, self).__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.temperature = float(temperature)
        self.weight_by_temperature = bool(weight_by_temperature)
        self.eps = eps

    def __call__(self, probs):
        """
        Args:
            probs: already-softmaxed probabilities, shape (..., num_classes)

        Returns:
            entropy over the last dimension.
            If weight_by_temperature=True:
                T * H(temperature_scale(probs, T))
            else:
                H(temperature_scale(probs, T))
        """
        probs_clamped = probs.clamp(min=self.eps)

        # Equivalent to softmax(log(probs) / T)
        if self.temperature != 1.0:
            log_probs = probs_clamped.log() / self.temperature
            probs_t = torch.softmax(log_probs, dim=-1)
        else:
            probs_t = probs_clamped / probs_clamped.sum(dim=-1, keepdim=True)

        probs_t = probs_t.clamp(min=self.eps)
        entropy = -(probs_t * probs_t.log()).sum(dim=-1)

        if self.weight_by_temperature:
            entropy = entropy * self.temperature

        return entropy


        
class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


class AugCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(AugCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_aug, x_ema):
        return -(1-self.alpha) * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
                  - self.alpha * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)


class GeneralizedCrossEntropy(nn.Module):
    """ Paper: https://arxiv.org/abs/1805.07836 """
    def __init__(self, q=0.8):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def __call__(self, logits, targets=None):
        probs = logits.softmax(1)
        if targets is None:
            targets = probs.argmax(dim=1)
        probs_with_correct_idx = probs.index_select(-1, targets).diag()
        return (1.0 - probs_with_correct_idx ** self.q) / self.q
