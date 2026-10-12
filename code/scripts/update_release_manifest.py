"""Record reviewed entrypoint/config changes without changing the frozen scientific core."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / 'reference_manifest.json'
    manifest = json.loads(path.read_text())
    for item in manifest['frozen_files']:
        if digest(ROOT / item['path']) != item['sha256']:
            raise SystemExit(f"Frozen scientific source changed: {item['path']}")
    if digest(ROOT / 'mira/_reference_core.py') != manifest['source_core_sha256']:
        raise SystemExit('Frozen MIRA core changed; this requires a new scientific reference.')
    entrypoints = [*ROOT.glob('mira/*.py'), *ROOT.glob('scripts/*.py'),
                   *ROOT.glob('scripts/*.sh'), ROOT / 'run_all.sh', ROOT / 'pyproject.toml',
                   ROOT / 'requirements-reference.txt', ROOT / 'configs/vit_profiles.json']
    manifest['api_and_entrypoint_files'] = [
        {'path': str(p.relative_to(ROOT)), 'sha256': digest(p)}
        for p in sorted(set(entrypoints))]
    manifest['config_files'] = [
        {'path': str(p.relative_to(ROOT)), 'sha256': digest(p)}
        for p in sorted((ROOT / 'configs').rglob('*.json'))
        if p.name != 'vit_profiles.json']
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Recorded {len(entrypoints)} entrypoints and {len(manifest['config_files'])} configs; frozen source verified.")


if __name__ == '__main__':
    main()
