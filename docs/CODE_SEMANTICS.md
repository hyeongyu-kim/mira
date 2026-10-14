# MIRA settings

| Setting | Default |
|---|---|
| Base method | TENT |
| Objective | Weighted HE + γ·EH |
| HE | Entropy of mean prediction |
| EH | Mean predictive entropy |
| MI | HE − EH |
| α | 0.03 |
| ρ | 0.90 |
| τ | 0.25 |
| γ | 1 |
| K | 300 |
| Perturbation | Classifier weight and bias; Gaussian |
| Scale | Feature-wise He-in |
| Feature-factor clipping | [0.25, 4] |
| Candidate noise | Shared across radii |
| Sample weights | Detached; unit mean |
| CNN adaptation parameters | Normalization weight and bias |
| ViT adaptation parameters | LayerNorm weight and bias |
| Main gradient normalization | Disabled |
| Gradient-norm controls | Global L2 norm = 1 |

## σ grid

```text
[1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2,
 1e-1, 3e-1, 5e-1, 1, 2, 3, 4, 5]
```

[Run configs](../code/configs/) · [Source manifest](../code/reference_manifest.json) ·
[Evaluation settings](SETUP.md#evaluation-settings)
