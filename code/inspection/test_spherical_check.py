"""Deterministic contract tests: geometry, standalone CLI, sibling outputs, preservation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from spherical_check import check_panorama, perspective, sample_erp


def erp_array(w=256,h=128):
    lon=((np.arange(w)+.5)/w-.5)*2*np.pi
    lat=(.5-(np.arange(h)+.5)/h)*np.pi
    x=np.cos(lat)[:,None]*np.sin(lon)[None,:]
    y=np.broadcast_to(np.sin(lat)[:,None],x.shape)
    z=np.cos(lat)[:,None]*np.cos(lon)[None,:]
    return np.rint(np.stack([x,y,z],axis=-1)*127.5+127.5).astype(np.uint8)

class GeometryTests(unittest.TestCase):
    def test_wrap_bilinear(self):
        im=np.array([[[0,0,0],[50,0,0],[100,0,0],[200,0,0]]])
        for x in (-.5,3.5):
            np.testing.assert_allclose(sample_erp(im,np.array([x]),np.array([0.])),[[100,0,0]])

    def test_center_and_pole_directions(self):
        for yaw,pitch,expected in [(0,0,[127.5,127.5,255]),(90,0,[255,127.5,127.5]),(180,0,[127.5,127.5,0]),(0,90,[127.5,255,127.5]),(0,-90,[127.5,0,127.5])]:
            np.testing.assert_allclose(perspective(erp_array(),yaw,pitch,90,1)[0,0],expected,atol=2)

    def test_continuous_wrap_view(self):
        view=perspective(erp_array(),180,0,90,64)
        self.assertTrue(np.isfinite(view).all())
        self.assertLess(float(np.abs(np.diff(view[32],axis=0)).max()),6)

class OutputContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.image=self.root/'room panorama.png'
        Image.fromarray(erp_array()).save(self.image)
        self.original=self.image.read_bytes()

    def tearDown(self): self.tmp.cleanup()

    def test_standalone_sibling_folder_and_hash_manifest(self):
        out=check_panorama(self.image,size=64)
        self.assertEqual(out,self.root/'room panorama_spherical_check')
        self.assertEqual(self.image.read_bytes(),self.original)
        manifest=json.loads((out/'manifest.json').read_text())
        self.assertEqual(manifest['input']['sha256'],hashlib.sha256(self.original).hexdigest())
        self.assertEqual(manifest['review_status'],'pending_manual_review')
        self.assertEqual(len(list(out.glob('*.png'))),10)
        self.assertIn('not a 3D', (out/'review.md').read_text())
        for item in manifest['outputs']:
            p=out/item['path']
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['sha256'])
            if p.suffix.lower() in ('.png','.jpg'):
                with Image.open(p) as im: im.load()

    def test_default_eight_yaws_sorted_then_poles(self):
        out=check_panorama(self.image,size=33,fov=75)
        names=[f'{i:02d}_yaw_{yaw:03d}.png' for i,yaw in enumerate(range(0,360,45))]+['08_zenith.png','09_nadir.png']
        self.assertEqual(sorted(p.name for p in out.glob('*.png')),names)
        m=json.loads((out/'manifest.json').read_text())
        self.assertEqual(m['config']['layout'],'eight')
        self.assertEqual(m['config']['fov_degrees'],75)
        self.assertEqual([v['yaw'] for v in m['views'][:8]],list(range(0,360,45)))
        self.assertEqual([v['pitch'] for v in m['views']], [0]*8+[90,-90])
        self.assertEqual([v['name']+'.png' for v in m['views']],names)
        self.assertEqual(m['config']['fov_horizontal_degrees'],75)
        self.assertEqual(m['config']['fov_vertical_degrees'],75)
        for i,yaw in enumerate(range(0,360,45)):
            with Image.open(out/names[i]) as im: pixel=np.array(im)[16,16]
            expected=np.array([np.sin(np.deg2rad(yaw)),0,np.cos(np.deg2rad(yaw))])*127.5+127.5
            np.testing.assert_allclose(pixel,expected,atol=2)
        with Image.open(out/'spherical_views.jpg') as im:self.assertEqual(im.size,(33*4,(33+28)*3))

    def test_legacy_six_layout(self):
        out=check_panorama(self.image,size=32,layout='six')
        self.assertEqual(sorted(p.stem for p in out.glob('*.png')),sorted(['front','right','back_seam','left','zenith','nadir']))
        m=json.loads((out/'manifest.json').read_text())
        self.assertEqual(m['config']['layout'],'six')
        self.assertEqual(len(m['views']),6)

    def test_invalid_layout_before_output_creation(self):
        with self.assertRaises(ValueError):check_panorama(self.image,size=32,layout='unknown')
        self.assertFalse(list(self.root.glob('*spherical_check*')))

    def test_cli_eight_default_and_explicit_six(self):
        script=Path(__file__).with_name('spherical_check.py')
        for extra,count in [([],10),(['--layout','six'],6)]:
            proc=subprocess.run([sys.executable,str(script),'--image',str(self.image),'--size','32',*extra],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            out=Path(json.loads(proc.stdout)['output_dir'])
            self.assertEqual(len(list(out.glob('*.png'))),count)
        proc=subprocess.run([sys.executable,str(script),'--image',str(self.image),'--layout','invalid'],capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)

    def test_repeat_preserves_existing_results_and_review(self):
        first=check_panorama(self.image,size=32)
        (first/'review.md').write_text('Human assessment: preserve this.')
        snapshots={p.name:p.read_bytes() for p in first.iterdir()}
        second=check_panorama(self.image,size=32)
        self.assertNotEqual(first,second)
        self.assertEqual(second.parent,self.root)
        self.assertTrue(second.name.startswith('room panorama_spherical_check_'))
        for name,content in snapshots.items(): self.assertEqual((first/name).read_bytes(),content)

    def test_optional_source_comparisons(self):
        source=self.root/'source.jpg'
        Image.new('RGB',(80,60),'red').save(source)
        before=source.read_bytes()
        out=check_panorama(self.image,source=source,size=32)
        self.assertTrue((out/'source_vs_front.jpg').is_file())
        self.assertTrue((out/'source_vs_erp.jpg').is_file())
        self.assertEqual(source.read_bytes(),before)

    def test_invalid_shape_does_not_create_output(self):
        invalid=self.root/'square.png'
        Image.new('RGB',(64,64)).save(invalid)
        with self.assertRaises(ValueError): check_panorama(invalid,size=32)
        self.assertFalse(list(self.root.glob('*spherical_check*')))

    def test_invalid_options_and_missing_source_do_not_create_output(self):
        with self.assertRaises(ValueError): check_panorama(self.image,size=0)
        with self.assertRaises(ValueError): check_panorama(self.image,size=32,fov=180)
        with self.assertRaises(FileNotFoundError): check_panorama(self.image,source=self.root/'missing.jpg',size=32)
        self.assertFalse(list(self.root.glob('*spherical_check*')))

    def test_cli_from_unrelated_cwd_with_space_in_path(self):
        script=Path(__file__).with_name('spherical_check.py')
        proc=subprocess.run([sys.executable,str(script),'--image',str(self.image),'--size','32'],cwd=self.root,capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        result=json.loads(proc.stdout)
        out=Path(result['output_dir'])
        self.assertEqual(out.parent,self.root)
        self.assertTrue((out/'spherical_views.jpg').is_file())

if __name__=='__main__': unittest.main()
