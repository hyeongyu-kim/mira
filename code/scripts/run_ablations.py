"""Run grouped CNN objective, construction and sensitivity configurations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = []
    for path in sorted((ROOT / 'configs').rglob('*.json')):
        config = json.loads(path.read_text())
        if 'run_id' not in config or config.get('backbone', '').startswith('vit_'):
            continue
        jobs.append(path)
    if not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / 'scripts/check_assets.py'), '--check-data'], check=True)
    for path in jobs:
        command = [sys.executable, str(ROOT / 'scripts/run_reference.py'), '--config', str(path)]
        if args.dry_run:
            print(' '.join(command))
        else:
            subprocess.run(command, check=True)
    print(f"{'Selected' if args.dry_run else 'Completed'} {len(jobs)} CNN configurations")


if __name__ == '__main__':
    main()
