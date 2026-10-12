"""Report pinned package and image-decoder compatibility with the measured reference."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-differences', action='store_true')
    args = parser.parse_args()
    differences = []
    packages = {}
    for line in (ROOT / 'requirements-reference.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = 'missing'
        packages[name] = actual
        if actual.split('+')[0] != expected:
            differences.append(f'{name}: expected {expected}, found {actual}')
    import torch
    from PIL import features
    decoder = {'jpeg': features.version('jpg'), 'libjpeg_turbo': features.version('libjpeg_turbo')}
    if platform.python_version_tuple()[:2] != ('3', '12'):
        differences.append('Reference Python major/minor is 3.12.')
    if torch.version.cuda != '12.1':
        differences.append(f'Reference torch CUDA build is 12.1; found {torch.version.cuda}.')
    if decoder != {'jpeg': '9.0', 'libjpeg_turbo': None}:
        differences.append('JPEG decoder differs from the conda reference (libjpeg 9, no turbo backend).')
    print(json.dumps({'python': platform.python_version(), 'packages': packages,
                      'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                      'image_decoder': decoder, 'differences': differences}, indent=2))
    if differences and not args.allow_differences:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
