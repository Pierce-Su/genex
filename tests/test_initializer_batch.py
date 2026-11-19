"""CPU-only tests. Fake inference is deliberate: no weights or CUDA required."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

INFERENCE = Path(__file__).resolve().parents[1] / 'code' / 'inference'
sys.path.insert(0, str(INFERENCE))


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for variant in ['photorealistic', 'stylized']:
            (self.root / variant).mkdir()
            Image.new('RGB', (80, 60), 'red').save(self.root / variant / '0005.jpg')
        self.data = {'samples': [{'index': 5, 'photorealistic': {
            'filename': '0005.jpg', 'prompt': 'photo caption', 'style': 'photorealistic'},
            'stylized': {'filename': '0005.jpg', 'prompt': 'painted caption', 'style': 'ukiyo-e'}}]}
        self.meta = self.root / 'metadata.json'
        self.meta.write_text(json.dumps(self.data))
        self.out = self.root / 'out'

    def batch(self):
        import initializer_batch
        return initializer_batch

    def args(self, *extra):
        return self.batch().parser().parse_args([
            '--input-root', str(self.root), '--output-root', str(self.out), *extra])

    def test_feature_exists(self):
        self.assertTrue((INFERENCE / 'initializer_batch.py').is_file(), 'batch CLI is missing')

    def test_plan_order_prompts_and_selection(self):
        b = self.batch()
        jobs = b.build_plan(self.args())
        self.assertEqual([j['key'] for j in jobs], ['photorealistic_0005', 'stylized_0005'])
        self.assertEqual([j['prompt'] for j in jobs], ['photo caption', 'painted caption'])
        self.assertEqual(jobs[0]['crop_box'], [10, 0, 70, 60])
        selected = b.build_plan(self.args('--select', 'stylized_0005', '--limit', '1'))
        self.assertEqual(selected, jobs[1:])
        self.assertEqual(b.build_plan(self.args()), jobs)
        self.assertFalse(self.out.exists())

    def fake_loader(self, args):
        self.loads += 1
        owner = self
        class Pipe:
            # Exact native signature: accidentally forwarding generator/steps fails.
            def __call__(self, image, prompt=None, guidance_scale=3.5):
                owner.calls.append((prompt, guidance_scale))
                if owner.should_fail and len(owner.calls) == 2:
                    raise RuntimeError('CUDA out of memory')
                return image.crop((10, 0, 70, 60)).resize((512, 512)), types.SimpleNamespace(
                    images=[Image.new('RGB', (2048, 1024), 'blue')])
        return Pipe()

    def run_fake(self, *extra):
        self.loads = 0
        self.calls = []
        self.seeds = []
        return self.batch().run(self.args(*extra), loader=self.fake_loader, seed_driver=self.seeds.append)

    def test_single_loader_native_boundary_and_resume(self):
        self.should_fail = False
        self.assertEqual(self.run_fake(), 0)
        self.assertEqual(self.loads, 1)
        self.assertEqual(self.calls, [('photo caption', 3.5), ('painted caption', 3.5)])
        self.assertEqual(len(set(self.seeds)), 2)
        manifests = [json.loads(p.read_text()) for p in sorted(self.out.glob('*/manifest.json'))]
        self.assertEqual([m['identity']['prompt'] for m in manifests], ['photo caption', 'painted caption'])
        self.assertEqual(self.run_fake('--resume'), 0)
        self.assertEqual(self.loads, 0)
        with self.assertRaisesRegex(ValueError, 'resume'):
            self.run_fake()
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_fake('--resume', '--seed', '999')
        self.assertEqual(self.loads, 0)

    def test_resume_rejects_tamper_and_source_change(self):
        self.should_fail = False
        self.run_fake()
        raw = self.out / 'photorealistic_0005' / 'rawERP.png'
        original = raw.read_bytes()
        raw.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.run_fake('--resume')
        raw.write_bytes(original)
        Image.new('RGB', (80, 60), 'green').save(self.root / 'stylized' / '0005.jpg')
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_fake('--resume')
        self.assertEqual(self.loads, 0)

    def test_fail_fast_and_resume_completed_only(self):
        self.should_fail = True
        self.assertEqual(self.run_fake(), 1)
        self.assertEqual(self.loads, 1)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue((self.out / 'photorealistic_0005' / 'manifest.json').exists())
        self.assertFalse((self.out / 'stylized_0005').exists())
        reports = list(self.out.glob('run-*.json'))
        self.assertEqual(json.loads(reports[0].read_text())['failed_key'], 'stylized_0005')
        self.should_fail = False
        self.assertEqual(self.run_fake('--resume'), 0)
        self.assertEqual(self.loads, 1)
        self.assertEqual(self.calls, [('painted caption', 3.5)])

    def test_fail_fast_does_not_attempt_third_sample(self):
        Image.new('RGB', (80, 60), 'green').save(self.root / 'stylized' / '0006.jpg')
        self.data['samples'].insert(0, {'index': 6, 'stylized': {
            'filename': '0006.jpg', 'prompt': 'third caption'}})
        self.meta.write_text(json.dumps(self.data))
        self.should_fail = True
        self.assertEqual(self.run_fake(), 1)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.out / 'stylized_0006').exists())
        jobs = self.batch().build_plan(self.args())
        self.assertEqual([j['index'] for j in jobs], [5, 5, 6])

    def test_plan_never_calls_loader_or_creates_outputs(self):
        from contextlib import redirect_stdout
        from io import StringIO
        output = StringIO()
        def forbidden(*args):
            raise AssertionError('plan must never call model loader or RNG')
        with redirect_stdout(output):
            self.assertEqual(self.batch().run(self.args('--plan'), loader=forbidden, seed_driver=forbidden), 0)
        self.assertEqual(json.loads(output.getvalue())['count'], 2)
        self.assertFalse(self.out.exists())

    def test_output_lock_prevents_second_writer(self):
        with self.batch().output_lock(self.out):
            with self.assertRaisesRegex(ValueError, 'lock'):
                with self.batch().output_lock(self.out):
                    self.fail('second writer acquired lock')

    def test_prompt_change_resume_rejected(self):
        self.should_fail = False
        self.run_fake()
        self.data['samples'][0]['stylized']['prompt'] = 'different caption'
        self.meta.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_fake('--resume')
        self.assertEqual(self.loads, 0)

    def test_interrupt_records_failure_and_stops(self):
        def interrupted(args):
            raise KeyboardInterrupt()
        self.assertEqual(self.batch().run(self.args(), loader=interrupted), 1)
        report = json.loads(next(self.out.glob('run-*.json')).read_text())
        self.assertIn('KeyboardInterrupt', report['error'])
        self.assertEqual(report['completed'], [])

    def test_metadata_validation_before_loader(self):
        self.data['samples'].append(self.data['samples'][0])
        self.meta.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.batch().build_plan(self.args())
        self.data['samples'] = self.data['samples'][:1]
        self.data['samples'][0]['stylized']['filename'] = '../outside.jpg'
        self.meta.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, 'filename'):
            self.batch().build_plan(self.args())


class NativeBoundaryTests(unittest.TestCase):
    def test_loader_pins_revision_and_never_loads_explorer(self):
        import initializer_batch as b
        from unittest.mock import Mock
        pipe = Mock()
        initializer = Mock()
        initializer.from_pretrained.return_value = pipe
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), bfloat16='bf16')
        args = argparse.Namespace(device='cuda', cpu_offload=False)
        with patch.dict(sys.modules, {'torch': torch, 'initializer': types.SimpleNamespace(
                GenExWorldInitializerPipeline=initializer)}):
            self.assertIs(b.load_pipeline(args), pipe)
        initializer.from_pretrained.assert_called_once_with(b.MODEL, revision=b.REVISION, torch_dtype='bf16')
        pipe.to.assert_called_once_with('cuda')
        pipe.enable_model_cpu_offload.assert_not_called()

    def test_global_seed_driver_seeds_cpu_and_all_cuda_rngs(self):
        import initializer_batch as b
        from unittest.mock import Mock
        torch = types.SimpleNamespace(manual_seed=Mock(), cuda=types.SimpleNamespace(
            is_available=lambda: True, manual_seed_all=Mock()))
        with patch.dict(sys.modules, {'torch': torch}):
            b.seed_global(123456)
        torch.manual_seed.assert_called_once_with(123456)
        torch.cuda.manual_seed_all.assert_called_once_with(123456)
        self.assertEqual(np.random.randint(100000), np.random.RandomState(123456).randint(100000))

    def test_unmodified_native_call_trigger_size_and_default_steps(self):
        class Fill:
            def __call__(self, *, prompt, image, mask_image, guidance_scale,
                         width, height, num_inference_steps=50, generator=None):
                return locals()
        with patch.dict(sys.modules, {'diffusers': types.SimpleNamespace(FluxFillPipeline=Fill)}):
            spec = importlib.util.spec_from_file_location('native_initializer', INFERENCE / 'initializer.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        pipe = module.GenExWorldInitializerPipeline()
        pipe.preprocess_image = lambda image: ('front', 'sparse')
        pipe.create_mask = lambda: 'mask'
        front, result = pipe(image='image', prompt='exact caption', guidance_scale=3.5)
        self.assertEqual(front, 'front')
        self.assertEqual(result['prompt'], 'GenEx Panoramic World Initialization, exact caption')
        self.assertEqual((result['width'], result['height'], result['num_inference_steps']), (2048, 1024, 50))
        self.assertIsNone(result['generator'])
        with self.assertRaises(TypeError):
            pipe(image='image', generator='not supported')


if __name__ == '__main__':
    unittest.main()
