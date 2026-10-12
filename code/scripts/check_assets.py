"""Check the exact scientific source/checkpoints and required local data before runs."""
from pathlib import Path
import argparse, hashlib, json, os

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--check-data',action='store_true')
ap.add_argument('--scope',choices=['main','vit','all'],default='all')
ap.add_argument('--vit-profile',help='ViT profile to check; defaults to the selected registry profile')
args=ap.parse_args()
manifest=json.loads((ROOT/'reference_manifest.json').read_text())
for item in manifest['frozen_files']+manifest.get('api_and_entrypoint_files',[])+manifest.get('config_files',[]):
    p=ROOT/item['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        raise SystemExit(f"Scientific reference source missing or changed: {p}")
core=ROOT/'mira/_reference_core.py'
assert hashlib.sha256(core.read_bytes()).hexdigest()==manifest['source_core_sha256']
checkpoint_root=Path(os.environ['MIRA_CHECKPOINT_ROOT'])
missing=[]
vit_files=set()
if args.scope in {'vit','all'}:
    registry=json.loads((ROOT/'configs/vit_profiles.json').read_text())
    profile_name=args.vit_profile or registry['default_profile']
    if profile_name not in registry['profiles']:ap.error('Unknown ViT profile: '+profile_name)
    profile=registry['profiles'][profile_name]
    for method in ['tent','full']:
        for seed in profile['default_seeds']:
            config=json.loads((ROOT/profile['configs'][method][str(seed)]).read_text())
            vit_files.add(Path(config['checkpoint']).name)
for item in manifest['checkpoints']:
    if args.scope=='main' and Path(item['file']).name not in {'Hendrycks2020AugMix_ResNeXt.pt','resnet50-0676ba61.pth'}:
        continue
    if args.scope=='vit' and Path(item['file']).name not in vit_files:
        continue
    if args.scope=='all' and item.get('architecture','').startswith('vit_') and Path(item['file']).name not in vit_files:
        continue
    p=checkpoint_root/item['file']
    if not p.is_file():missing.append(str(p));continue
    if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        raise SystemExit(f"Checkpoint SHA256 mismatch: {p}")
if missing:raise SystemExit('Missing source checkpoints:\n'+'\n'.join(missing))
if args.check_data:
    data=Path(os.environ['MIRA_DATA_ROOT'])
    corruptions=json.loads((ROOT/'configs/c100_full_s1.json').read_text())['corruptions']
    cifar_datasets=[] if args.scope=='vit' else ['CIFAR-100-C'] if args.scope=='main' else ['CIFAR-10-C','CIFAR-100-C']
    for dataset in cifar_datasets:
        for name in ['labels',*corruptions]:
            if not (data/dataset/(name+'.npy')).is_file():missing.append(str(data/dataset/(name+'.npy')))
    for name in corruptions:
        if not (data/'ImageNet-C'/name/'5').is_dir():missing.append(str(data/'ImageNet-C'/name/'5'))
    if missing:raise SystemExit('Missing evaluation data:\n'+'\n'.join(missing))
print('PASS: exact reference source and source checkpoints'+(', required data paths' if args.check_data else ''))
