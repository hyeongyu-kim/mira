"""Ensure a profile switch cannot silently mix measured seeds or previous run caches."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_vit', ROOT / 'scripts/run_vit.py')
run_vit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_vit)


class ProfileSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((ROOT / 'configs/vit_profiles.json').read_text())

    def test_version_switch_preserves_distinct_run_ids_and_measured_seeds(self):
        current, jobs = run_vit.selected_configs(self.registry, 'vit_b16_batch64_v1')
        self.assertEqual(current, 'vit_b16_batch64_v1')
        self.assertEqual([(m, s) for m, s, _ in jobs], [('tent', 1), ('full', 1)])
        current_ids = {json.loads((ROOT / p).read_text())['run_id'] for _, _, p in jobs}
        _, previous = run_vit.selected_configs(self.registry, 'vit_b16_batch128_previous')
        self.assertEqual({s for _, s, _ in previous}, {1, 2, 3})
        previous_ids = {json.loads((ROOT / p).read_text())['run_id'] for _, _, p in previous}
        self.assertFalse(current_ids & previous_ids)
        self.assertEqual({json.loads((ROOT / p).read_text())['batch_size'] for _, _, p in jobs}, {64})
        self.assertEqual({json.loads((ROOT / p).read_text())['batch_size'] for _, _, p in previous}, {128})
        switched = deepcopy(self.registry)
        switched['default_profile'] = 'vit_b16_batch128_previous'
        name, selected = run_vit.selected_configs(switched)
        self.assertEqual(name, 'vit_b16_batch128_previous')
        self.assertEqual(selected, previous)

    def test_explicit_three_seeds_are_opt_in_and_reject_duplicates(self):
        _, jobs = run_vit.selected_configs(self.registry, 'vit_b16_batch64_v1', seeds=[1, 2, 3])
        self.assertEqual(len(jobs), 6)
        with self.assertRaises(ValueError):
            run_vit.selected_configs(self.registry, seeds=[1, 1])
        with self.assertRaises(ValueError):
            run_vit.selected_configs(self.registry, seeds=[4])
        with self.assertRaises(ValueError):
            run_vit.selected_configs(self.registry, profile_name='unknown')


if __name__ == '__main__':
    unittest.main()
