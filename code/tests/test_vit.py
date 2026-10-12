"""Check actual transformer feature capture and LayerNorm-only updates."""
from copy import deepcopy
import unittest

import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from mira import MIRA, MIRAConfig


class ReferenceObjectiveMIRA(MIRA):
    def _objective(self, betta):
        return self._betta_objective(betta['HE'], betta['EH'])


class ViTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = VisionTransformer(img_size=32, patch_size=16, embed_dim=24,
                                      depth=2, num_heads=3, mlp_ratio=2, num_classes=5)
        self.images = torch.randn(4, 3, 32, 32)
        self.config = MIRAConfig(K=12, sigma_grid=(.01, .1, 1.))

    def test_feature_cache_is_cls_token_classifier_input(self):
        adapter = MIRA(self.model, self.config)
        model = adapter.model.base
        with torch.no_grad():
            expected = model.forward_head(model.forward_features(self.images), pre_logits=True)
            captured = adapter.model.extract_last_features(self.images)
        self.assertEqual(tuple(captured.shape), (4, 24))
        self.assertTrue(torch.equal(expected, captured))
        self.assertIs(adapter.model.last_layer, model.head)

    def test_only_layernorm_updates_and_classifier_and_reset_are_exact(self):
        adapter = MIRA(self.model, self.config)
        original = deepcopy(adapter.model.state_dict())
        norm_ids = {id(p) for module in adapter.model.modules() if isinstance(module, nn.LayerNorm)
                    for p in module.parameters(recurse=False)}
        self.assertEqual({id(p) for p in adapter.params}, norm_ids)
        probabilities = adapter.predict_proba(self.images)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(4)))
        self.assertFalse(adapter.last_weights.requires_grad)
        self.assertTrue(any(not torch.equal(original[k], v) for k, v in adapter.model.state_dict().items()))
        allowed = {name for name, p in adapter.model.named_parameters() if id(p) in norm_ids}
        for name, value in adapter.model.state_dict().items():
            if name not in allowed:
                self.assertTrue(torch.equal(original[name], value), name)
        adapter.reset()
        self.assertTrue(all(torch.equal(v, adapter.model.state_dict()[k]) for k, v in original.items()))
        self.assertEqual(adapter.optimizer.state_dict()['state'], {})

    def test_vit_prediction_and_update_match_preserved_reference(self):
        public = MIRA(deepcopy(self.model), self.config)
        reference = ReferenceObjectiveMIRA(deepcopy(self.model), self.config)
        torch.manual_seed(29)
        observed = public(self.images)
        torch.manual_seed(29)
        expected = reference(self.images)
        self.assertTrue(torch.equal(observed, expected))
        for name, value in public.model.state_dict().items():
            self.assertTrue(torch.equal(value, reference.model.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
