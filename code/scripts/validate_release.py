"""Offline checks for source integrity, portable profiles and reported statistics."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    manifest = json.loads((ROOT / 'reference_manifest.json').read_text())
    verified = set()
    for section in ['frozen_files', 'api_and_entrypoint_files', 'config_files']:
        for entry in manifest.get(section, []):
            path = ROOT / entry['path']
            assert path.is_file(), f"Missing source: {entry['path']}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], f"Changed source: {entry['path']}"
            verified.add(entry['path'])
    assert hashlib.sha256((ROOT / 'mira/_reference_core.py').read_bytes()).hexdigest() == manifest['source_core_sha256']
    configs = {}
    for path in (ROOT / 'configs').rglob('*.json'):
        config = json.loads(path.read_text())
        if 'run_id' not in config:
            continue
        assert config['run_id'] not in configs, f"Duplicate result ID: {config['run_id']}"
        assert '${MIRA_DATA_ROOT}' == config['data_dir']
        assert config['checkpoint'].startswith('${MIRA_CHECKPOINT_ROOT}/')
        assert config['stream_seed'] == config['seed'] == config['perturbation_seed']
        assert config['steps'] == 1 and config['severity'] == 5
        configs[config['run_id']] = config
    registry = json.loads((ROOT / 'configs/vit_profiles.json').read_text())
    assert registry['default_profile'] in registry['profiles']
    for profile in registry['profiles'].values():
        for method in ['tent', 'full']:
            for seed in profile['available_seeds']:
                config = json.loads((ROOT / profile['configs'][method][str(seed)]).read_text())
                assert config['seed'] == seed
                assert config['loss'] == method
        if profile.get('reference_results'):
            measured = read_csv(ROOT / profile['reference_results'])
            assert {int(r['seed']) for r in measured} == set(profile['default_seeds'])
            assert len(measured) == 2 * len(profile['default_seeds'])
            for row in measured:
                method = 'tent' if row['method'] == 'tent' else 'full'
                config = json.loads((ROOT / profile['configs'][method][row['seed']]).read_text())
                assert row['run_id'] == config['run_id']
                assert int(row['batch_size']) == config['batch_size']
                assert row['backbone'] == config['backbone']
                assert float(row['lr']) == config['lr']
                assert row['protocol'] == config['protocol']
                assert row['checkpoint_sha256'] == config['checkpoint_sha256']
                assert row['domains_completed'] == '15'
                printed_metrics = profile.get('reported_seed_metrics', {}).get(row['seed'], {}).get(method, {})
                for metric, printed in printed_metrics.items():
                    precision = len(printed.split('.')[1])
                    assert f"{float(row[metric]):.{precision}f}" == printed
    commands = json.loads((ROOT / 'docs/table_commands.json').read_text())
    for table, paths in commands.items():
        for path in paths:
            assert (ROOT / path).is_file(), (table, path)
    assert commands['3 ViT'] == [], 'Historical Table 3 has no verified exact reproduction profile.'
    for entry in json.loads((ROOT/'results/diagnostics/index.json').read_text()):
        archive = ROOT/entry['archive']
        assert archive.is_file(), 'Missing diagnostics archive: '+entry['archive']
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == entry['sha256']
    raw = read_csv(ROOT / 'results/seed_results.csv')
    summaries = read_csv(ROOT / 'results/mean_sd_summary.csv')
    keys = list(summaries[0])[:list(summaries[0]).index('n_seeds')]
    groups = defaultdict(list)
    for row in raw:
        assert row['run_id'] in configs
        groups[tuple(row[k] for k in keys)].append(row)
    for row in summaries:
        runs = groups[tuple(row[k] for k in keys)]
        assert len(runs) == int(row['n_seeds'])
        assert len({r['seed'] for r in runs}) == len(runs)
        for metric in ['accuracy_percent', 'mmc', 'ece', 'nll']:
            values = [float(r[metric]) for r in runs]
            assert all(math.isfinite(v) for v in values)
            assert abs(statistics.mean(values) - float(row[metric + '_mean'])) < 1e-10
            if len(values) > 1:
                assert abs(statistics.stdev(values) - float(row[metric + '_sd'])) < 1e-10
            else:
                assert row[metric + '_sd'] == ''
    for profile in registry['profiles'].values():
        seed_path = ROOT / profile['reference_results']
        measured = read_csv(seed_path)
        for row in read_csv(seed_path.with_name('vit_mean_sd_summary.csv')):
            method = 'tent' if row['method'] == 'TENT' else 'tent_laplace_betta'
            runs = [r for r in measured if r['method'] == method]
            assert int(row['n_seeds']) == len(runs)
            for metric in ['accuracy_percent', 'mmc', 'ece', 'nll']:
                values = [float(r[metric]) for r in runs]
                assert abs(statistics.mean(values) - float(row[metric + '_mean'])) < 1e-10
                if len(values) == 1:
                    assert row[metric + '_sd'] == ''
                else:
                    assert abs(statistics.stdev(values) - float(row[metric + '_sd'])) < 1e-10
    audit = json.loads((ROOT.parent / 'docs/paper_numeric_audit.json').read_text())
    assert audit['all_supported_numeric_cells_match']
    assert all(r['match'] == 'True' for r in read_csv(ROOT.parent / 'docs/paper_numeric_audit.csv'))
    print(json.dumps({'status': 'passed', 'verified_source_and_config_files': len(verified),
                      'portable_run_configs': len(configs), 'archived_seed_runs': len(raw),
                      'recomputed_scientific_groups': len(summaries),
                      'matched_paper_numeric_cells': audit['matched_numeric_cells'],
                      'default_vit_profile': registry['default_profile'],
                      'measured_vit_seeds': registry['profiles'][registry['default_profile']]['default_seeds']}, indent=2))


if __name__ == '__main__':
    main()
