"""CPU-only inspection adapter; never writes into the generation tree."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import time

from PIL import Image
from spherical_check import check_panorama, digest


def inventory(root, expected, keys=None):
    root = Path(root).resolve(strict=True)
    manifests = sorted(root.glob('*/manifest.json'))
    if keys:
        manifests = [p for p in manifests if p.parent.name in keys]
        if {p.parent.name for p in manifests} != set(keys):
            raise ValueError('Requested sample keys are missing')
    if len(manifests) != expected:
        raise ValueError(f'Expected {expected} completed samples, found {len(manifests)}')
    records = []
    for path in manifests:
        m = json.loads(path.read_text())
        key = path.parent.name
        if m['identity']['key'] != key:
            raise ValueError(f'Identity mismatch: {path}')
        artifacts = m['artifacts']
        source = 'source' + Path(m['identity']['source_relative']).suffix.lower()
        if set(artifacts) != {'rawERP.png', 'front.png', source}:
            raise ValueError(f'Unexpected artifacts: {key}')
        for name, sha in artifacts.items():
            p = path.parent / name
            if digest(p) != sha:
                raise ValueError(f'Artifact hash mismatch: {p}')
            with Image.open(p) as im:
                im.load()
                if name in ('rawERP.png', 'front.png'):
                    expected_size = (2048, 1024) if name == 'rawERP.png' else (512, 512)
                    if im.mode != 'RGB' or im.size != expected_size:
                        raise ValueError(f'Invalid native artifact mode/size: {p}')
        if artifacts[source] != m['identity']['source_sha256']:
            raise ValueError(f'Source identity hash mismatch: {key}')
        records.append({'key': key, 'manifest': str(path), 'manifest_sha256': digest(path),
                        'artifacts': artifacts, 'source': source})
    return records


def run(root, output, expected=140, keys=None, size=512, exit_marker=None):
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    if output == root or root in output.parents:
        raise ValueError('Derived output must be outside generation tree')
    if exit_marker is not None and Path(exit_marker).read_text().strip() != '0':
        raise ValueError('Generation exit marker is not zero')
    started = time.monotonic()
    records = inventory(root, expected, keys)
    # Fail before any output write if inputs/count/options are invalid.
    if not 16 <= size <= 2048:
        raise ValueError('size must be in [16,2048]')
    output.mkdir(parents=True, exist_ok=True)
    from spherical_check import fresh_output
    run_dir = fresh_output(output / 'batch.png')
    (run_dir / 'inventory.json').write_text(json.dumps(records, indent=2))
    rendered = []
    for item in records:
        sample = run_dir / item['key']
        sample.mkdir()
        original = root / item['key']
        # Logical staging paths preserve skill sibling output semantics. Copies, not
        # symlinks, keep downloaded result folders usable after rental destruction.
        import shutil
        for name in ('rawERP.png', item['source']):
            shutil.copyfile(original / name, sample / name)
        t = time.monotonic()
        out = check_panorama(sample / 'rawERP.png', sample / item['source'], size=size)
        m = json.loads((out / 'manifest.json').read_text())
        if m['projection_status'] != 'rendered' or len(m['views']) != 10:
            raise ValueError(f'Incomplete projection: {out}')
        for image in m['outputs']:
            p = out / image['path']
            if digest(p) != image['sha256']:
                raise ValueError(f'Rendered hash mismatch: {p}')
            with Image.open(p) as im:
                im.load()
        elapsed = time.monotonic() - t
        row = {'key': item['key'], 'output': str(out.relative_to(run_dir)),
               'seconds': elapsed, 'review_status': 'pending_manual_review'}
        rendered.append(row)
        with (run_dir / 'rendered.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    # Independent second pass detects original edits during rendering.
    if inventory(root, expected, keys) != records:
        raise ValueError('Original inventory changed during inspection')
    counts = dict(Counter(x['key'].rsplit('_', 1)[0] for x in records))
    summary = {'input_root': str(root), 'expected': expected, 'rendered': len(rendered),
               'variants': counts, 'png_views': 10 * len(rendered),
               'seconds': time.monotonic() - started, 'cpu_only': True,
               'review_status': 'pending_manual_review',
               'limits': 'Same-center 2D reprojection only; not semantic acceptance or 3D validation.'}
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    rows = ['<!doctype html><meta charset="utf-8"><title>GenEx sphere inspection</title>',
            '<style>body{font:16px sans-serif;background:#202020;color:#eee}img{max-width:100%}a{color:#acf}</style>',
            '<h1>GenEx CPU spherical projections</h1><p>Yaw0 = ERP center; yaw180 = wrap seam; positive yaw = right. Eight horizontal headings plus zenith/nadir. All quality reviews pending unless review.md says otherwise. Not 3D validation.</p>']
    for row in rendered:
        link = html.escape(row['output'], quote=True)
        rows.append(f'<h2>{html.escape(row["key"])}</h2><a href="{link}/manifest.json">Manifest</a> | <a href="{link}/review.md">Review</a><br><img loading="lazy" src="{link}/spherical_views.jpg"><br><img loading="lazy" src="{link}/source_vs_front.jpg"><br><img loading="lazy" src="{link}/erp_preview.jpg">')
    (run_dir / 'index.html').write_text('\n'.join(rows))
    print(json.dumps({'run_dir': str(run_dir), **summary}), flush=True)
    return run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--expected', type=int, default=140)
    p.add_argument('--key', action='append', help='Explicit bounded sample subset')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--exit-marker', type=Path, help='Require this file to contain 0; never waits/polls')
    a = p.parse_args()
    run(a.input_root, a.output_root, a.expected, a.key, a.size, a.exit_marker)


if __name__ == '__main__':
    main()
