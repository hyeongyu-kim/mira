"""Compare executable MIRA quantities with the paper's equations independently."""
from copy import deepcopy
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from mira import MIRA, MIRAConfig


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(5)
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(self.norm(x))


class PaperEquationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.adapter = MIRA(Toy(), MIRAConfig(K=4, gamma=.5))
        self.features = torch.tensor([[.1, .2, 1., 4., 20.], [.2, .4, 2., 8., 40.]])

    def test_featurewise_he_in_factors_match_appendix_d4(self):
        s0 = math.sqrt(2 / self.features.shape[1])
        activation = self.features.abs().mean(0)
        feature_factor = (activation / activation.mean()).clamp(.25, 4.)
        weight_scale, bias_scale = self.adapter.model._noise_scales(self.features)
        expected = (s0 * feature_factor)[None].expand_as(self.adapter.model.last_layer.weight)
        self.assertTrue(torch.allclose(weight_scale, expected, atol=1e-7))
        self.assertAlmostEqual(float(bias_scale), s0, places=6)

    def test_cached_logits_equal_explicit_perturbed_classifiers(self):
        W = self.adapter.model.last_layer.weight.detach().clone()
        b = self.adapter.model.last_layer.bias.detach().clone()
        s0 = math.sqrt(2 / W.shape[1])
        activation = self.features.abs().mean(0)
        factors = (activation / activation.mean()).clamp(.25, 4.)
        torch.manual_seed(31)
        epsilon_w = torch.randn(4, *W.shape)
        epsilon_b = torch.randn(4, b.numel())
        scales = torch.tensor([.03, .3, 1.])
        expected = torch.stack([torch.stack([
            F.linear(self.features, W + sigma * s0 * factors * epsilon_w[k],
                     b + sigma * s0 * epsilon_b[k]) for k in range(4)]) for sigma in scales])
        torch.manual_seed(31)
        actual = self.adapter.model.mc_logits_grid_from_features(self.features, scales, 4, same_noise=True)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-6))
        self.assertTrue(torch.equal(W, self.adapter.model.last_layer.weight))
        self.assertTrue(torch.equal(b, self.adapter.model.last_layer.bias))

    def test_mc_probabilities_entropies_match_section_4_1(self):
        logits = torch.tensor([[[1., 2., 3.], [4., 2., 1.]],
                               [[3., 1., 2.], [1., 3., 2.]],
                               [[2., 3., 1.], [2., 1., 3.]],
                               [[1., 3., 2.], [3., 2., 1.]]])
        probabilities = logits.softmax(-1)
        mean = probabilities.mean(0)
        H = -(mean * mean.log()).sum(-1)
        E = -(probabilities * probabilities.log()).sum(-1).mean(0)
        output, actual_mean, actual_H, actual_E, _ = self.adapter._posterior_from_mc_logits(logits)
        self.assertTrue(torch.allclose(actual_mean, mean))
        self.assertTrue(torch.allclose(actual_H, H))
        self.assertTrue(torch.allclose(actual_E, E))
        self.assertTrue(torch.allclose(output.softmax(-1), mean))

    def test_detached_mi_confidence_weighted_loss_matches_section_4_3(self):
        H = torch.tensor([1.7, 1.1, 1.3, .9], requires_grad=True)
        E = torch.tensor([1.5, 1.0, .6, .4], requires_grad=True)
        probabilities = torch.tensor([[.7, .2, .1], [.1, .8, .1],
                                      [.5, .3, .2], [.05, .05, .9]], requires_grad=True)
        mi = H.detach() - E.detach()
        w_mi = torch.exp(-(mi - mi.mean()) / .25)
        w_mi = w_mi / w_mi.mean()
        confidence = probabilities.detach().max(-1).values
        w_conf = confidence / confidence.mean()
        weights = w_mi * w_conf
        weights = weights / weights.mean()
        expected = (weights * (H + .5 * E)).mean()
        actual = self.adapter._objective({'HE': H, 'EH': E, 'p_mean': probabilities})
        self.assertTrue(torch.allclose(actual, expected))
        self.assertFalse(self.adapter.last_weights.requires_grad)
        actual.backward()
        self.assertTrue(torch.allclose(H.grad, weights / 4))
        self.assertTrue(torch.allclose(E.grad, .5 * weights / 4))
        self.assertIsNone(probabilities.grad)


if __name__ == '__main__':
    unittest.main()
