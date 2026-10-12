# Versioned ViT reference profiles

The selected public reference is `vit_b16_torchvision_batch64_v2`: torchvision `vit_b_16`,
checkpoint `vit_b_16-c867db91.pth`, batch size 64, SGD learning rate 0.00025,
one update per batch, 15 severity-5 ImageNet-C corruptions, and reset at each corruption.
TENT and TENT+MIRA each have one completed seed (seed 1). Seeds 2 and 3 have executable
configs but no measured results in this version. Single-seed SD is left blank.

The default lives in [code/configs/vit_profiles.json](../code/configs/vit_profiles.json).
The previous timm batch64 and batch128 validations have separate profiles, results and run IDs.
The selected seed1 TENT/MIRA accuracies are 41.64/44.76%, with ECE .1167/.0549 and
NLL 3.0582/2.8406. Its mean/std is [.485,.456,.406]/[.229,.224,.225]; the preserved timm
profiles use mean/std [.5,.5,.5]. Checkpoint/preprocessing and learning-rate differences stay explicit.

| Profile | Backbone | LR | Batch | Measured seeds |
| --- | --- | ---: | ---: | --- |
| `vit_b16_torchvision_batch64_v2` (default) | torchvision ViT-B/16 V1 | .00025 | 64 | 1 |
| `vit_b16_batch64_v1` | timm orig_in21k_ft_in1k | .001 | 64 | 1 |
| `vit_b16_batch128_previous` | timm orig_in21k_ft_in1k | .001 | 128 | 1,2,3 |

```bash
cd code
python scripts/run_vit.py --list-profiles
bash scripts/run_vit.sh --dry-run
bash scripts/run_vit.sh                         # batch64, seed1, both methods
bash scripts/run_vit.sh --seeds 1 2 3           # opt-in additional measurements
bash scripts/run_vit.sh --methods full          # MIRA only
bash scripts/run_vit.sh --profile vit_b16_batch64_v1
bash scripts/run_vit.sh --profile vit_b16_batch128_previous
```

For a future reference change:

1. Copy the current paired configs to a new versioned directory, keeping the old files.
2. Give every changed run a new `run_id`, `protocol` and `release_profile`; set its scientific
   settings and environment overrides consistently. Keep corresponding stream and perturbation
   seeds equal across methods. Never reuse old adapted state or result IDs.
3. Add the config paths to the profile registry. Set `reference_results` to that version's
   exported seed CSV and, optionally, `reported_seed_metrics` to its displayed values.
   Use only measured seeds as its `default_seeds`; `available_seeds` can include planned
   additional repeats.
4. Run both methods, export seed results, and inspect per-corruption results and diagnostics.
5. Select the new `default_profile` only when its paired results have been verified. Update the
   README and results note with the actual number of measured seeds.
6. Record the reviewed files and validate them:

```bash
python scripts/update_release_manifest.py
python scripts/validate_release.py
python -m unittest discover -s tests -v
python scripts/run_vit.py --dry-run
```

The manifest updater verifies every frozen scientific file before recording config and
entrypoint changes. It refuses to relabel a modified MIRA core as the old scientific reference.
The runner rejects changed configs when reusing completed results. Completed corruption
boundaries retain RNG states; a partially written corruption is replayed from its source reset.

Historical paper Table 3 reports 42.03/45.17% accuracy. Recovered notes establish LR0.00025
and batch64, but not checkpoint/seed/complete stream. The selected torchvision reconstruction
is lower by 0.39/0.41 pp; the gain +3.12 pp is close to the paper's +3.14 pp.
All versions retain their measured results without claiming exact original provenance.
The selected Full run uses the preserved MI-first fallback in 99.62% of batches;
this profile does not demonstrate universal joint feasibility.
