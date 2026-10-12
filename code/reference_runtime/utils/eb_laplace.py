## 특정값 고정해서 쓰고, multi sigma 가능한거

# models/laplace_wrapper.py
# (Simplified + multi-sigma MC) LLLA-style last-layer wrapper
# - 베이스 모델 구조는 그대로 유지 (파라미터 안 건드림)
# - 마지막 nn.Linear 레이어만 찾아서, 그 앞 feature를 훅으로 캡처
# - forward: self.base(x) (deterministic)
# - forward_lastlayer_mc_logits:
#       단일 sigma에 대해 K개 샘플 -> [K,B,C]
# - forward_lastlayer_mc_logits_multi_sigma:
#       sigma_list에 대해 각 sigma마다 K개 샘플 -> [S,K,B,C]

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


__all__ = ["LaplaceWrapper"]


def _find_last_linear_module(model: nn.Module) -> nn.Linear:
    """모델 안에서 마지막 nn.Linear 레이어 하나 찾아서 리턴."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("LaplaceWrapper: no nn.Linear found as last layer.")
    return last


class LaplaceWrapper(nn.Module):
    """
    아주 단순한 last-layer MC 샘플링 래퍼.

    - base_model: 원래 분류 모델 (nn.Module)
    - __init__ 에서 하는 일:
        * base_model 를 self.base 에 저장
        * 마지막 nn.Linear 를 찾아 self.last_layer 로 저장
    - forward(x):
        * 그냥 self.base(x) 호출 (deterministic)
    - forward_lastlayer_mc_logits(x, K, sigma):
        * sigma 하나에 대해 마지막 레이어 weight를
          W_k = W + eps * (sigma * |W|)
          형태로 K번 샘플링 → logits [K,B,C]
    - forward_lastlayer_mc_logits_multi_sigma(x, sigma_list, K):
        * sigma_list 의 각 sigma_s 에 대해 위와 같은 샘플링
          → logits [S, K, B, C]
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.last_layer: nn.Linear = _find_last_linear_module(self.base)

    # ------------------- 기본 forward -------------------
    def forward(self, x):
        """일반 inference: 그냥 base 모델을 그대로 사용."""
        return self.base(x)

    # ------------------- feature capture -------------------
    @contextmanager
    def _capture_last_features(self):
        """
        마지막 Linear 들어가기 직전의 feature를 훅으로 캡처.
        """
        if self.last_layer is None:
            raise RuntimeError("LaplaceWrapper: last_layer is None.")

        feat_container = {"feat": None}

        def pre_hook(module, inputs):
            # inputs: (feat, ) 꼴이라고 가정
            feat_container["feat"] = inputs[0]
            return None

        handle = self.last_layer.register_forward_pre_hook(pre_hook, with_kwargs=False)
        try:
            yield feat_container
        finally:
            handle.remove()

    # ------------------- 단일 sigma 기준 MC -------------------
    def forward_lastlayer_mc_logits(self, x, K: int = 20, sigma: float = 1.0):
        """
        하나의 sigma 에 대해 마지막 레이어 weight를 K번 샘플링해서
        logits [K,B,C] 를 반환.

        - sigma: 스칼라 scale. 실제 std는 sigma * |W| 로 사용.
        """
        if self.last_layer is None:
            raise RuntimeError("LaplaceWrapper: last_layer is None.")

        # 1) backbone 한 번 돌려서 feature 캡처
        with self._capture_last_features() as cap:
            _ = self.base(x)          # [B,C] (혹은 뭔가 출력이 나올 것) → 중요 X
            feats = cap["feat"]       # [B,D]
            if feats is None:
                raise RuntimeError("LaplaceWrapper: Failed to capture last-layer features.")
        B = feats.shape[0]

        # 2) last layer 파라미터
        W = self.last_layer.weight   # [C,D]
        device = W.device
        dtype = W.dtype
        C, D = W.shape

        # std_w = sigma * |W|
        sigma_val = float(sigma)

        ############################## HERE ##########################################
        ############################## HERE ##########################################
        # 안정적인 posterior scale.
        # unit noise(std=sigma)는 너무 큼.
        # |W| element-wise는 너무 작은 weight에서 noise가 거의 0이 됨.
        # class-wise RMS scale을 기본 std로 사용.
        w_rms = W.detach().pow(2).mean(dim=1, keepdim=True).sqrt()  # [C,1]
        w_floor = W.detach().pow(2).mean().sqrt().clamp_min(1e-6) * 0.05
        base_std_w = w_rms.clamp_min(w_floor).expand_as(W)          # [C,D]

        std_w = sigma_val * base_std_w                              # [C,D]
        ############################## HERE ##########################################
        ############################## HERE ##########################################



        K_int = int(K)
        # epsW: [K,C,D]
        epsW = torch.randn((K_int, C, D), device=device, dtype=dtype)
        # W_k: [K,C,D]
        W_k = W.unsqueeze(0) + epsW * std_w

        # bias
        if self.last_layer.bias is not None:
            b = self.last_layer.bias  # [C]
            # 최소 수정: bias noise는 끔. weight noise만으로 충분.
            b_k = b.unsqueeze(0).expand(K_int, C)
        else:
            b_k = None



        # 3) logits: feats [B,D], W_k [K,C,D] → [K,B,C]
        logits = torch.einsum("bd,kcd->kbc", feats, W_k)
        if b_k is not None:
            logits = logits + b_k.unsqueeze(1)  # [K,1,C] → [K,B,C]

        return logits

    def forward_lastlayer_mc_probs(self, x, K: int = 20, sigma: float = 1.0):
        """
        단일 sigma 기준 MC logits → softmax 확률 [K,B,C]
        """
        logits = self.forward_lastlayer_mc_logits(x, K=K, sigma=sigma)
        return F.softmax(logits, dim=-1)

    # ------------------- multi-sigma MC -------------------
    def forward_lastlayer_mc_logits_multi_sigma(self, x, sigma_list, K: int = 20):
        """
        sigma_list 에 있는 여러 sigma_s 에 대해,
        마지막 레이어 weight를 각각 K번 샘플링하여
        logits [S,K,B,C] 를 반환.

        - x: 입력 [B,...]
        - sigma_list: iterable of float 또는 1D tensor (길이 S)
        - K: 각 sigma 당 샘플 개수
        """
        if self.last_layer is None:
            raise RuntimeError("LaplaceWrapper: last_layer is None.")

        # 1) backbone 한 번만 돌려서 feature 캡처
        with self._capture_last_features() as cap:
            _ = self.base(x)
            feats = cap["feat"]       # [B,D]
            if feats is None:
                raise RuntimeError("LaplaceWrapper: Failed to capture last-layer features.")
        B = feats.shape[0]

        # 2) last layer 파라미터
        W = self.last_layer.weight   # [C,D]
        device = W.device
        dtype = W.dtype
        C, D = W.shape

        # sigma_list 텐서화
        sigma_tensor = torch.as_tensor(sigma_list, device=device, dtype=dtype)
        if sigma_tensor.ndim != 1:
            sigma_tensor = sigma_tensor.view(-1)
        S = sigma_tensor.shape[0]
        K_int = int(K)

        ############################## HERE ##########################################
        ############################## HERE ##########################################
        # unit std는 너무 큼.
        # class-wise RMS weight scale을 posterior base std로 사용.
        w_rms = W.detach().pow(2).mean(dim=1, keepdim=True).sqrt()  # [C,1]
        w_floor = W.detach().pow(2).mean().sqrt().clamp_min(1e-6) * 0.05
        base_std_w = w_rms.clamp_min(w_floor).expand_as(W).view(1, 1, C, D)
        ############################## HERE ##########################################
        ############################## HERE ##########################################


        # sigma별 std: [S,1,1,1] * [1,1,C,D] -> [S,1,C,D]
        # std_w_SKCD = sigma_tensor.view(S, 1, 1, 1) * base_std_w
        std_w_SKCD = sigma_tensor.view(S, 1, 1, 1) * base_std_w

        # weight 샘플링: epsW [S,K,C,D]
        epsW = torch.randn((S, K_int, C, D), device=device, dtype=dtype)
        # W_mu broadcast: [1,1,C,D]
        W_mu_SKCD = W.view(1, 1, C, D)
        # W_SKCD: [S,K,C,D]
        W_SKCD = W_mu_SKCD + epsW * std_w_SKCD





        # bias (있으면)
        if self.last_layer.bias is not None:
            b = self.last_layer.bias  # [C]
            # 최소 수정: bias noise는 끔.
            b_SKC = b.view(1, 1, C).expand(S, K_int, C)
        else:
            b_SKC = None

        # 3) logits: feats [B,D], W_SKCD [S,K,C,D] → [S,K,B,C]
        logits = torch.einsum("bd,skcd->skbc", feats, W_SKCD)
        if b_SKC is not None:
            logits = logits + b_SKC.unsqueeze(2)  # [S,K,1,C] → [S,K,B,C]

        return logits

    def forward_lastlayer_mc_probs_multi_sigma(self, x, sigma_list, K: int = 20):
        """
        multi-sigma logits → softmax 확률 [S,K,B,C]
        """
        logits = self.forward_lastlayer_mc_logits_multi_sigma(x, sigma_list, K=K)
        return F.softmax(logits, dim=-1)
