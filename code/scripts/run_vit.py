"""Run a versioned ViT reference profile; the default profile is selected in JSON."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def selected_configs(registry, profile_name=None, seeds=None, methods=None):
    name = profile_name or registry['default_profile']
    if name not in registry['profiles']:
        raise ValueError(f"Unknown profile {name!r}; choose from {list(registry['profiles'])}")
    profile = registry['profiles'][name]
    selected_seeds = profile['default_seeds'] if seeds is None else seeds
    selected_methods = ['tent', 'full'] if methods is None else methods
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError('Each seed may be requested once.')
    if any(seed not in profile['available_seeds'] for seed in selected_seeds):
        raise ValueError(f"Available seeds: {profile['available_seeds']}")
    return name, [(method, seed, profile['configs'][method][str(seed)])
                  for seed in selected_seeds for method in selected_methods]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', help='Default: configs/vit_profiles.json default_profile')
    ap.add_argument('--seeds', type=int, nargs='+', help='Default: measured seeds for the selected profile')
    ap.add_argument('--methods', choices=['tent', 'full'], nargs='+')
    ap.add_argument('--max-batches', type=int, default=0, help='A separate smoke run; never a full reference result')
    ap.add_argument('--list-profiles', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='Display selected commands without loading models')
    args = ap.parse_args()
    registry = json.loads((ROOT / 'configs/vit_profiles.json').read_text())
    if args.list_profiles:
        print(json.dumps(registry, indent=2))
        return
    try:
        name, jobs = selected_configs(registry, args.profile, args.seeds, args.methods)
    except ValueError as error:
        ap.error(str(error))
    commands = [[sys.executable, str(ROOT / 'scripts/run_reference.py'), '--config', str(ROOT / config)]
                + (['--max-batches', str(args.max_batches)] if args.max_batches else [])
                for method, seed, config in jobs]
    print(json.dumps({'profile': name, 'runs': [dict(method=m, seed=s, config=c)
                                               for m, s, c in jobs]}, indent=2), flush=True)
    if args.dry_run:
        for command in commands:
            print(' '.join(command))
        return
    for var in ['MIRA_DATA_ROOT', 'MIRA_CHECKPOINT_ROOT', 'CUDA_VISIBLE_DEVICES']:
        if not os.environ.get(var):
            ap.error(f'Set {var} before evaluation.')
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_assets.py'), '--scope', 'vit', '--vit-profile', name, '--check-data'], check=True)
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
