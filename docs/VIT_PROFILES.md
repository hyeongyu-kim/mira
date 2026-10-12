# Versioned ViT reference profiles

The selected public reference is `vit_b16_batch64_v1`: batch size 64, SGD learning rate
0.001, one update per batch, 15 severity-5 ImageNet-C corruptions, and reset at each corruption.
TENT and TENT+MIRA each have one completed seed (seed 1). Seeds 2 and 3 have executable
configs but no measured results in this version. Single-seed SD is left blank.

The default lives in [code/configs/vit_profiles.json](../code/configs/vit_profiles.json).
The previous batch-128 validation has a separate profile, result directory and run IDs.
Switching profiles does not overwrite or pool the two protocols.

```bash
cd code
python scripts/run_vit.py --list-profiles
bash scripts/run_vit.sh --dry-run
bash scripts/run_vit.sh                         # batch64, seed1, both methods
bash scripts/run_vit.sh --seeds 1 2 3           # opt-in additional measurements
bash scripts/run_vit.sh --methods full          # MIRA only
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

Historical paper Table 3 reports 42.03/45.17% accuracy. Its exact original checkpoint/config
and complete logs have not been recovered. Neither versioned release profile is represented
as exact reproduction of those historical rows.
