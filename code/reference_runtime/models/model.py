"""Model dispatch for the public CNN/ViT TENT and MIRA recipes only.

The runner installs strict local torchvision/timm checkpoint loaders before this
factory is called. RobustBench source files must already exist and are verified
by the runner; no alternate weights or architectures are substituted.
"""
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from pathlib import Path


def get_torchvision_model(model_name, weight_version='IMAGENET1K_V1'):
    raise RuntimeError('Install the explicit local checkpoint loader before creating a model')


def get_timm_model(model_name):
    raise RuntimeError('Install the explicit local checkpoint loader before creating a model')


def get_model(cfg, num_classes, device):
    arch = cfg.MODEL.ARCH
    if arch in {'resnet34', 'resnet50', 'vit_b_16'}:
        model, preprocess = get_torchvision_model(arch, weight_version=cfg.MODEL.WEIGHTS)
    elif arch == 'vit_base_patch16_224.orig_in21k_ft_in1k':
        model, preprocess = get_timm_model(arch)
    elif arch in {'Hendrycks2020AugMix_ResNeXt', 'Hendrycks2020AugMix_WRN'}:
        dataset = cfg.CORRUPTION.DATASET.split('_')[0]
        expected = Path(cfg.CKPT_DIR)/dataset/'corruptions'/(arch+'.pt')
        if not expected.is_file():
            raise FileNotFoundError(f'Required local source checkpoint: {expected}')
        if expected.resolve() != globals().get('_explicit_checkpoint'):
            raise ValueError('RobustBench source path differs from the verified checkpoint')
        model = load_model(arch, cfg.CKPT_DIR, dataset, ThreatModel.corruptions)
        preprocess = None
    else:
        raise ValueError(f'Architecture outside the public CNN/ViT recipes: {arch}')
    return model.to(device), preprocess
