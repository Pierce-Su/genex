"""Adapter tests use native-size analytic input, never inference."""
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from spherical_check import digest
from batch_spherical import inventory, run


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.sample = self.inputs / 'photorealistic_0000'
        self.sample.mkdir(parents=True)
        for name, size in [('rawERP.png', (2048, 1024)), ('front.png', (512, 512)), ('source.jpg', (80, 60))]:
            Image.new('RGB', size, 'green').save(self.sample / name)
        self.m = {'identity': {'key': 'photorealistic_0000', 'source_relative': 'photo/a.jpg'}, 'artifacts': {p.name: digest(p) for p in self.sample.iterdir()}}
        self.m['identity']['source_sha256'] = self.m['artifacts']['source.jpg']
        (self.sample / 'manifest.json').write_text(json.dumps(self.m))

    def tearDown(self):
        self.tmp.cleanup()

    def test_count_and_corrupt_hash_fail_before_write(self):
        with self.assertRaises(ValueError):
            inventory(self.inputs, 140)
        self.m['artifacts']['rawERP.png'] = '0' * 64
        (self.sample / 'manifest.json').write_text(json.dumps(self.m))
        with self.assertRaises(ValueError):
            run(self.inputs, self.root / 'out', 1, size=32)
        self.assertFalse((self.root / 'out').exists())

    def test_external_fresh_results_and_source_preservation(self):
        before = {p.name: digest(p) for p in self.sample.iterdir()}
        first = run(self.inputs, self.root / 'out', 1, size=32)
        second = run(self.inputs, self.root / 'out', 1, size=32)
        self.assertNotEqual(first, second)
        self.assertEqual(before, {p.name: digest(p) for p in self.sample.iterdir()})
        self.assertEqual(json.loads((first / 'summary.json').read_text())['png_views'], 10)
        self.assertIn('photorealistic_0000', (first / 'index.html').read_text())

    def test_generation_tree_and_exit_marker_guards(self):
        with self.assertRaises(ValueError):
            run(self.inputs, self.inputs / 'bad', 1)
        marker = self.root / 'exit'
        marker.write_text('1\n')
        with self.assertRaises(ValueError):
            run(self.inputs, self.root / 'out', 1, exit_marker=marker)
        self.assertFalse((self.root / 'out').exists())

    def test_missing_subset_and_artifact_identity(self):
        with self.assertRaises(ValueError):
            inventory(self.inputs, 1, ['missing'])
        self.m['identity']['key'] = 'wrong'
        (self.sample / 'manifest.json').write_text(json.dumps(self.m))
        with self.assertRaises(ValueError):
            inventory(self.inputs, 1)


if __name__ == '__main__':
    unittest.main()
