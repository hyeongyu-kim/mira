"""Use explicitly supplied source checkpoints instead of downloading replacements."""
from pathlib import Path


def load_source_classifier(architecture, checkpoint, weight_version='IMAGENET1K_V1'):
    """Return a strictly loaded torchvision/timm model with input normalization.

    The second return value is the source preprocessing transform, excluding
    normalization. ImageNet-C images are already 224x224; the reference protocol
    uses ToTensor directly and the normalization included in this model.
    """
    from collections import OrderedDict
    from types import SimpleNamespace
    import torch
    from torch import nn
    import torchvision

    class ImageNormalizer(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

        def forward(self, images):
            return (images - self.mean) / self.std

    def normalize(model, mean, std):
        return nn.Sequential(OrderedDict([
            ('normalize', ImageNormalizer(mean, std)), ('model', model),
        ]))

    factory = SimpleNamespace(normalize_model=normalize)
    install_local_torchvision_loader(factory, checkpoint)
    if architecture in torchvision.models.list_models(module=torchvision.models):
        return factory.get_torchvision_model(architecture, weight_version)
    return factory.get_timm_model(architecture)


def install_local_torchvision_loader(model_module, checkpoint):
    """Keep torchvision/timm architecture metadata and strictly load a local file.

    No pretrained=True calls or network access are used. The timm path is needed
    for the archived ViT-B/16 orig_in21k_ft_in1k architecture.
    """
    import torch
    import torchvision
    from torchvision import transforms
    model_module._explicit_checkpoint = Path(checkpoint).resolve()

    def load(model_name, weight_version="IMAGENET1K_V1"):
        available = torchvision.models.list_models(module=torchvision.models)
        if model_name not in available:
            raise ValueError(f"Not a torchvision architecture: {model_name}")
        weights = getattr(torchvision.models.get_model_weights(model_name), weight_version)
        model = torchvision.models.get_model(model_name, weights=None)
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        transform = weights.transforms()
        model = model_module.normalize_model(model, transform.mean, transform.std)
        preprocess = transforms.Compose([
            transforms.Resize(transform.resize_size, interpolation=transform.interpolation),
            transforms.CenterCrop(transform.crop_size), transforms.ToTensor(),
        ])
        return model, preprocess

    model_module.get_torchvision_model = load

    def load_timm(model_name):
        import timm
        if model_name not in timm.list_models(pretrained=True):
            raise ValueError(f"Not a pretrained timm architecture: {model_name}")
        timm.layers.set_fused_attn(True)
        model = timm.create_model(model_name, pretrained=False)
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config)
        for transform in preprocess.transforms[::-1]:
            if isinstance(transform, transforms.Normalize):
                model = model_module.normalize_model(model, transform.mean, transform.std)
                preprocess.transforms.remove(transform)
                break
        return model, preprocess

    model_module.get_timm_model = load_timm
