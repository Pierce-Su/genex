#!/usr/bin/env python3
"""Sequential curated_set -> raw ERP. No torch/diffusers import in plan mode."""
import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time
import uuid

from PIL import Image, ImageOps

MODEL = 'genex-world/World-Initializer-image-to-panorama'
REVISION = 'bdf9a02257c4c2728bcd90404d53687a9641e205'
UPSTREAM = 'eda604588d0a205c57e2b2599eec566d97162c2e'
VARIANTS = ('photorealistic', 'stylized')
PACKAGES = ('torch', 'diffusers', 'transformers', 'accelerate', 'numpy', 'Pillow',
            'sentencepiece', 'safetensors', 'huggingface-hub')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def versions():
    result = {'python': platform.python_version()}
    for package in PACKAGES:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True,
                   help='curated_set with photorealistic/ and stylized/ subdirectories')
    p.add_argument('--metadata', type=Path, help='defaults to INPUT_ROOT/metadata.json')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--plan', '--dry-run', action='store_true', dest='plan',
                   help='validate selected source images and print JSON; no output writes or model imports')
    p.add_argument('--select', action='append', default=[], metavar='VARIANT_STEM',
                   help='exact key, e.g. stylized_0005; repeatable')
    p.add_argument('--limit', type=int)
    p.add_argument('--seed', type=int, default=0, help='base seed; per-key SHA256 derived seed')
    p.add_argument('--guidance-scale', type=float, default=3.5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-offload', action='store_true', help='opt-in model CPU offload to cuda:0')
    return p


def normalized_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def build_plan(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if not 0 <= args.seed < 2**63:
        raise ValueError('--seed must be in [0, 2**63)')
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        raise ValueError('--guidance-scale must be finite and nonnegative')
    if args.cpu_offload and args.device != 'cuda':
        raise ValueError('--cpu-offload requires --device cuda (GPU 0)')
    root = args.input_root.resolve()
    data = json.loads((args.metadata or root / 'metadata.json').read_text(encoding='utf-8'))
    if not isinstance(data.get('samples'), list) or not data['samples']:
        raise ValueError('metadata requires a nonempty samples list')
    entries = {}
    for sample in data['samples']:
        index = sample.get('index')
        if type(index) is not int or index < 0:
            raise ValueError('sample index must be a nonnegative integer')
        found = False
        for variant in VARIANTS:
            if variant not in sample:
                continue
            found = True
            value = sample[variant]
            filename = value.get('filename')
            if (not isinstance(filename, str) or not filename or
                    Path(filename).name != filename or '\\' in filename or
                    not Path(filename).stem.isascii() or not Path(filename).stem.isdigit()):
                raise ValueError('variant filename must be a numeric basename, e.g. 0005.jpg')
            if int(Path(filename).stem) != index:
                raise ValueError('filename stem must agree with sample index')
            key = f'{variant}_{Path(filename).stem}'
            if key in entries:
                raise ValueError(f'duplicate sample key: {key}')
            if not isinstance(value.get('prompt'), str) or not value['prompt'].strip():
                raise ValueError(f'{key}: missing exact variant prompt')
            entries[key] = (index, variant, filename, value, sample)
        if not found:
            raise ValueError(f'sample {index}: no supported variants')
    unknown = set(args.select) - entries.keys()
    if unknown:
        raise ValueError(f'unknown selection keys: {sorted(unknown)}')
    keys = sorted(entries, key=lambda key: (entries[key][0], entries[key][1], key))
    if args.select:
        keys = [key for key in keys if key in args.select]
    if args.limit:
        keys = keys[:args.limit]
    settings = {
        'model': MODEL, 'model_revision': REVISION, 'upstream_commit': UPSTREAM,
        'initializer_sha256': sha256(Path(__file__).with_name('initializer.py')),
        'driver_sha256': sha256(__file__), 'versions': versions(),
        'guidance_scale': args.guidance_scale, 'num_inference_steps': 50,
        'size': [2048, 1024], 'device': args.device, 'cpu_offload': args.cpu_offload,
        'dtype': 'bfloat16', 'seed_base': args.seed,
        'seed_policy': 'sha256(base:variant_stem) first 8 bytes big endian modulo 2**63',
        'normalization': 'PIL.ImageOps.exif_transpose then RGB',
        'crop': 'native center square, resize512 default PIL, cube front 90deg',
        'output': 'raw native output; no compositing or seam pass',
    }
    jobs = []
    for key in keys:
        index, variant, filename, value, sample = entries[key]
        relative = Path(variant) / filename
        source = (root / relative).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f'{key}: source escapes input root')
        image = normalized_image(source)
        w, h = image.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        jobs.append({
            'schema': 1, 'key': key, 'index': index, 'variant': variant,
            'variant_metadata': value,
            'sample_metadata': {k: v for k, v in sample.items() if k not in VARIANTS},
            'source_relative': relative.as_posix(), 'source_sha256': sha256(source),
            'normalized_size': [w, h], 'crop_box': [left, top, left + side, top + side],
            'prompt': value['prompt'],
            'effective_prompt': 'GenEx Panoramic World Initialization, ' + value['prompt'],
            'seed': int.from_bytes(hashlib.sha256(f'{args.seed}:{key}'.encode()).digest()[:8], 'big') % 2**63,
            'settings': settings,
        })
    return jobs


def validate_completed(directory, job, resume):
    if not directory.exists():
        return False
    if not resume:
        raise ValueError(f'{directory}: already exists; use --resume to validate, never overwrite')
    try:
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['identity'] != job:
            raise ValueError(f'{directory}: identity mismatch; use a fresh output root')
        expected = {'rawERP.png', 'front.png', 'source' + Path(job['source_relative']).suffix.lower()}
        if set(manifest['artifacts']) != expected:
            raise ValueError(f'{directory}: artifact list mismatch')
        for filename, digest in manifest['artifacts'].items():
            if sha256(directory / filename) != digest:
                raise ValueError(f'{directory / filename}: hash mismatch')
        for filename, size in [('rawERP.png', (2048, 1024)), ('front.png', (512, 512))]:
            with Image.open(directory / filename) as image:
                image.load()
                if image.size != size or image.mode != 'RGB':
                    raise ValueError(f'{directory / filename}: invalid size/mode')
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'{directory}: incomplete/corrupt output; move aside or use a fresh root: {exc}') from exc
    return True


def seed_global(seed):
    # Native wrapper has no generator kwarg. Reset all RNGs for each stable key.
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pipeline(args):
    import torch
    from initializer import GenExWorldInitializerPipeline
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA unavailable; plan mode is CPU-only, inference is not')
    pipe = GenExWorldInitializerPipeline.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=0)
    else:
        pipe.to(args.device)
    return pipe


@contextmanager
def output_lock(root):
    # Vast/Linux: forbid concurrent writers, including writers to different selections.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'{root}: another batch process holds the output lock') from exc
        yield


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def run(args, loader=load_pipeline, seed_driver=seed_global):
    jobs = build_plan(args)
    if args.plan:
        print(json.dumps({'count': len(jobs), 'jobs': jobs}, indent=2, ensure_ascii=False))
        return 0
    root = args.output_root.resolve()
    with output_lock(root):
        # Check every selected destination before any weight load or new inference.
        pending = [job for job in jobs if not validate_completed(root / job['key'], job, args.resume)]
        report = {'selected': len(jobs), 'skipped': len(jobs) - len(pending), 'completed': [],
                  'failed_key': None, 'error': None, 'started_unix': time.time()}
        report_path = root / f'run-{time.time_ns()}-{uuid.uuid4().hex[:8]}.json'
        stage = None
        current_key = None
        try:
            if pending:
                print(f'Loading initializer ONCE; {len(pending)} pending, {report["skipped"]} verified skips', flush=True)
                pipe = loader(args)
                for number, job in enumerate(pending, 1):
                    current_key = job['key']
                    print(f'[{number}/{len(pending)}] {current_key} seed={job["seed"]}', flush=True)
                    source = args.input_root / job['source_relative']
                    # Detect source edits between plan and inference.
                    if sha256(source) != job['source_sha256']:
                        raise ValueError(f'{current_key}: source changed after planning')
                    image = normalized_image(source)
                    seed_driver(job['seed'])
                    started = time.monotonic()
                    front, output = pipe(image=image, prompt=job['prompt'],
                                         guidance_scale=args.guidance_scale)
                    raw = output.images[0]
                    if raw.size != (2048, 1024) or raw.mode != 'RGB' or front.size != (512, 512):
                        raise ValueError('pipeline returned unexpected native output size/mode')
                    stage = Path(tempfile.mkdtemp(prefix=f'.partial-{current_key}-', dir=root))
                    source_name = 'source' + source.suffix.lower()
                    shutil.copyfile(source, stage / source_name)
                    if sha256(stage / source_name) != job['source_sha256']:
                        raise ValueError(f'{current_key}: source changed during inference')
                    front.save(stage / 'front.png')
                    raw.save(stage / 'rawERP.png')
                    artifacts = {name: sha256(stage / name) for name in [source_name, 'front.png', 'rawERP.png']}
                    write_json(stage / 'manifest.json', {
                        'identity': job, 'artifacts': artifacts,
                        'inference_seconds': time.monotonic() - started,
                    })
                    # All writes are staged. Never replace an existing sample directory.
                    destination = root / current_key
                    if destination.exists():
                        raise ValueError(f'{destination}: appeared during inference; refusing overwrite')
                    stage.rename(destination)
                    stage = None
                    report['completed'].append(current_key)
        except (Exception, KeyboardInterrupt) as exc:
            report['failed_key'] = current_key
            report['error'] = f'{type(exc).__name__}: {exc}'
            print(f'STOP: {report["error"]}. No further samples attempted. '
                  'After fixing the cause, start a NEW process with --resume; '
                  'never reuse a CUDA OOM/device-assert process.', file=sys.stderr, flush=True)
        finally:
            if stage is not None:
                report['partial_directory'] = str(stage)
            report['finished_unix'] = time.time()
            write_json(report_path, report)
        print(f'Summary: {len(report["completed"])} completed, {report["skipped"]} verified skips; '
              f'report={report_path}', flush=True)
        return 1 if report['error'] else 0


def main():
    try:
        return run(parser().parse_args())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
