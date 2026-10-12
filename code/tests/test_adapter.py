"""Scientific invariants and numerical equivalence to the preserved reference core."""
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mira import MIRA, MIRAConfig, ReliabilityProbe
from mira.metrics import evaluate


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,5,1)
        self.norm = nn.BatchNorm2d(5)
        self.fc = nn.Linear(5,3)

    def forward(self,x):
        return self.fc(self.norm(self.conv(x)).mean((2,3)))


class ReferenceObjectiveMIRA(MIRA):
    def _objective(self,betta):
        return self._betta_objective(betta['HE'],betta['EH'])


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = Toy()
        self.images = torch.randn(8,3,8,8)

    def test_default_prediction_and_update_match_reference_objective(self):
        cfg=MIRAConfig(K=24)
        a=MIRA(deepcopy(self.model),cfg)
        b=ReferenceObjectiveMIRA(deepcopy(self.model),cfg)
        torch.manual_seed(13);pa=a(self.images)
        torch.manual_seed(13);pb=b(self.images)
        self.assertTrue(torch.equal(pa,pb))
        for k,v in a.model.state_dict().items():
            self.assertTrue(torch.equal(v,b.model.state_dict()[k]),k)

    def test_classifier_frozen_weights_detached_and_reset_exact(self):
        adapter=MIRA(self.model,MIRAConfig(K=24))
        original=deepcopy(adapter.model.state_dict())
        classifier=adapter.model.last_layer.weight.detach().clone()
        adapter(self.images)
        self.assertTrue(torch.equal(classifier,adapter.model.last_layer.weight))
        self.assertFalse(adapter.last_weights.requires_grad)
        self.assertTrue(any(not torch.equal(original[k],v) for k,v in adapter.model.state_dict().items()))
        adapter.reset()
        self.assertTrue(all(torch.equal(v,adapter.model.state_dict()[k]) for k,v in original.items()))
        self.assertEqual(adapter.optimizer.state_dict()['state'],{})

    def test_eh_coefficient_does_not_change_confidence_weight_exponent(self):
        a=MIRA(deepcopy(self.model),MIRAConfig(K=24,gamma=1))
        b=MIRA(deepcopy(self.model),MIRAConfig(K=24,gamma=2))
        torch.manual_seed(13);a(self.images)
        torch.manual_seed(13);b(self.images)
        self.assertTrue(torch.equal(a.last_weights,b.last_weights))
        self.assertTrue(any(not torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items()))

    def test_actual_complete_gradient_norm_is_matched(self):
        a=MIRA(self.model,MIRAConfig(K=24,gradient_norm_matching=True))
        observed=[]
        original_step=a.optimizer.step
        def capture(*args,**kwargs):
            observed.append(float(sum(p.grad.detach().float().square().sum() for p in a.params if p.grad is not None).sqrt()))
            return original_step(*args,**kwargs)
        a.optimizer.step=capture
        a(self.images)
        self.assertAlmostEqual(observed[0],1.,places=5)

    def test_config_ignores_and_restores_ambient_research_flags(self):
        key='BETTA_POSTERIOR_SCALE_MODE';previous=os.environ.get(key)
        os.environ[key]='scalar'
        try:
            a=MIRA(self.model,MIRAConfig(K=24))
            self.assertEqual(a.model.posterior_scale_mode,'feature')
            self.assertEqual(os.environ[key],'scalar')
        finally:
            if previous is None:os.environ.pop(key,None)
            else:os.environ[key]=previous

    def test_ten_bin_metrics_include_confidence_one(self):
        p=torch.tensor([[1.,0.],[.7,.3],[.5,.5]])
        result=evaluate(p,torch.tensor([0,1,0]))
        self.assertAlmostEqual(result['accuracy_percent'],200/3,places=4)
        self.assertAlmostEqual(result['ece'],.4,places=6)
        self.assertAlmostEqual(result['nll'],(-math.log(.3)-math.log(.5))/3,places=6)

    def test_shared_grid_draws_and_fresh_loss_draws(self):
        a=MIRA(self.model,MIRAConfig(K=24))
        original=torch.randn
        draws=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            draws.append(result.detach().clone())
            return result
        with patch('torch.randn',side_effect=capture):a(self.images)
        self.assertEqual(len(draws),4)  # weight/bias once for grid, once for loss
        self.assertEqual(draws[0].shape,draws[2].shape)
        self.assertFalse(torch.equal(draws[0],draws[2]))

    def test_probe_keeps_base_trainable_set_and_classifier(self):
        self.model.eval()
        self.model.conv.requires_grad_(False)
        flags={n:p.requires_grad for n,p in self.model.named_parameters()}
        head=self.model.fc.weight.detach().clone()
        probe=ReliabilityProbe(self.model,MIRAConfig(K=24))
        weights,_=probe.weights(self.images)
        self.assertFalse(weights.requires_grad)
        self.assertAlmostEqual(float(weights.mean()),1.,places=6)
        self.assertEqual(flags,{n:p.requires_grad for n,p in self.model.named_parameters()})
        self.assertTrue(torch.equal(head,self.model.fc.weight))

    def test_existing_mi_first_fallback_without_agreement(self):
        a=MIRA(self.model,MIRAConfig(K=2,sigma_grid=(.1,1.,10.)))
        logits=torch.tensor([[[[10.,0.,-20.]],[[10.,0.,-20.]]],
                             [[[10.,0.,-20.]],[[0.,10.,-20.]]],
                             [[[2.,0.,-20.]],[[0.,2.,-20.]]]])
        a.model.mc_logits_grid_from_features=lambda *args,**kwargs:logits
        sigma,stats=a._select_sigma_from_features(torch.zeros(1,5))
        self.assertEqual(float(sigma),1.)
        self.assertEqual(stats['idx_anchor'],1)
        self.assertLess(float(stats['agreement'][1]),a.agreement_threshold)

    def test_feasible_interpolation_is_in_log_sigma(self):
        a=MIRA(self.model,MIRAConfig(K=2,sigma_grid=(.1,1.,10.)))
        logits=torch.tensor([[[[10.,0.,-20.]],[[10.,0.,-20.]]],
                             [[[10.,0.,-20.]],[[.1,0.,-20.]]],
                             [[[10.,0.,-20.]],[[0.,10.,-20.]]]])
        a.model.mc_logits_grid_from_features=lambda *args,**kwargs:logits
        sigma,stats=a._select_sigma_from_features(torch.zeros(1,5))
        self.assertGreater(float(sigma),.1)
        self.assertLess(float(sigma),1.)
        self.assertEqual(stats['sigma_interp_kind'],'linear_logsigma_mi_elbow_agreement')


if __name__=='__main__':unittest.main()
