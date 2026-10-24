"""Check the published saved fields and tables; no integrations or retiming.

Run from any directory: python /path/to/repository/examples/replay_paper.py
"""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def replay(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        target = (root/item['public_path']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Unsafe public path: '+item['public_path'])
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['public_sha256']:
            raise ValueError('Public artifact hash mismatch: '+item['public_path'])
        if item['transformation'].startswith('deterministic gzip'):
            if hashlib.sha256(gzip.decompress(raw)).hexdigest() != item['source_sha256']:
                raise ValueError('Decoded source hash mismatch: '+item['public_path'])
    with gzip.open(root/'data/baselines/common_cases.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    original = {}
    for name in ['points','bands_012','bands_345']:
        with gzip.open(root/f'data/frozen/{name}.jsonl.gz', 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['case_id'] in original:
                    raise ValueError('Duplicate original case ID')
                original[row['case_id']] = row
    if len(rows) != 5214 or len(original) != 5214:
        raise ValueError('Incomplete common reference set')
    summary = json.loads((root/'data/baselines/summary.json').read_text(encoding='utf-8'))
    methods = summary['methods']
    bands = defaultdict(list)
    points = []
    seen = set()
    for row in rows:
        cid = row['case_id']
        if not re.fullmatch(r'(?:points|bands):(?:0|[1-9][0-9]*):(?:0|[1-9][0-9]*)', cid):
            raise ValueError('Invalid literal case ID')
        if cid in seen or cid not in original or set(row['methods']) != set(methods):
            raise ValueError('Common-set identity mismatch')
        seen.add(cid)
        source = original[cid]
        for key in ['q','w','y','kappa_s','amplitude','reference','frequency_index','group_id']:
            if row[key] != source[key]:
                raise ValueError('Saved original/common field mismatch: '+cid+':'+key)
        if row['w'] != row['q']/row['y'] or not row['reference']['accepted']:
            raise ValueError('Invalid frequency or unaccepted reference')
        for name in ['cut_fixed','cut_controlled','cut_controlled_1e10','born3']:
            if row['methods'][name]['value'] != source['methods'][name]['value']:
                raise ValueError('Saved numerical-method value mismatch')
        if row['mode'] == 'bands':
            bands[row['group_id']].append(row)
        else:
            points.append(row)
    if len(points) != 408 or len(bands) != 6 or any(len(g) != 801 for g in bands.values()):
        raise ValueError('Incomplete points or complete bands')
    with (root/'data/baselines/band_metrics.csv').open(newline='', encoding='utf-8') as stream:
        expected = {(int(r['band_id']),r['method']):r for r in csv.DictReader(stream)}
    if len(expected) != 48:
        raise ValueError('Incomplete band metrics')
    passes = {name:[0,0] for name in methods}
    checked = 0
    for band, group in sorted(bands.items()):
        group.sort(key=lambda r:r['frequency_index'])
        w = np.array([r['w'] for r in group])
        if np.any(np.diff(w) <= 0):
            raise ValueError('Nonincreasing frequency grid')
        widths = np.diff(np.log(w))
        def norm(values):
            power = np.abs(values)**2
            return float(np.sqrt(np.sum(widths*(power[:-1]+power[1:])/2)))
        reference = np.array([complex(*r['reference']['value']) for r in group])
        sigma = np.array([r['reference']['uncertainty_empirical'] for r in group])
        amplitude = np.array([r['amplitude'] for r in group])
        denominator = norm(reference-amplitude)-norm(sigma)
        if denominator <= 0:
            raise ValueError('Unresolved residual denominator')
        for name in methods:
            values = np.array([complex(*r['methods'][name]['value']) for r in group])
            maximum = float(np.max(np.abs(values-reference)+sigma))
            rho = (norm(values-reference)+norm(sigma))/denominator
            saved = expected[band,name]
            for key,value in [('max_complex_error_with_uncertainty',maximum),
                              ('relative_residual_error_with_uncertainty',rho),
                              ('denominator_margin',denominator)]:
                if not np.isclose(value,float(saved[key]),rtol=2e-13,atol=2e-16):
                    raise ValueError('Band metric mismatch: '+str((band,name,key)))
            for index,target in enumerate([1e-8,1e-10]):
                passed = maximum <= target and rho <= .01
                key = 'pass_primary' if index == 0 else 'pass_secondary'
                if passed != (saved[key] == 'True'):
                    raise ValueError('Band decision mismatch')
                passes[name][index] += int(passed)
            checked += 1
    point_passes = {}
    expected_points = {r['method']:r for r in summary['point_metrics']}
    for name in methods:
        errors = [abs(complex(*r['methods'][name]['value'])-complex(*r['reference']['value']))
                  +r['reference']['uncertainty_empirical'] for r in points]
        counts = [sum(e <= target for e in errors) for target in [1e-8,1e-10]]
        saved = expected_points[name]
        if counts != [saved['primary_pass_count'],saved['secondary_pass_count']]:
            raise ValueError('Point decision mismatch')
        point_passes[name] = counts
    cold = json.loads((root/'data/timing/cold_confirmation.json').read_text(encoding='utf-8'))
    if len(cold['pairs']) != 16 or len(cold['records']) != 32:
        raise ValueError('Incomplete saved timing records')
    expected_pairs = {(band,pair) for band in range(1,5) for pair in range(4)}
    if {(p['band'],p['pair']) for p in cold['pairs']} != expected_pairs:
        raise ValueError('Saved timing pair identity mismatch')
    expected_records = {(band,pair,kind) for band,pair in expected_pairs for kind in ['glow','cut']}
    if {(r['band'],r['pair'],r['kind']) for r in cold['records']} != expected_records:
        raise ValueError('Saved timing record identity mismatch')
    if len(cold['band_summary']) != 4 or {s['band'] for s in cold['band_summary']} != {1,2,3,4}:
        raise ValueError('Saved timing band summary identity mismatch')
    medians = {}
    durations = {}
    for record in cold['records']:
        duration = record['prepare_s']+record['first_s']
        if not np.isclose(duration,record['cold_s'],rtol=1e-12,atol=1e-14):
            raise ValueError('Saved preparation-plus-first sum mismatch')
        for score in record['scores']:
            passed = score['max_with_margin'] <= 1e-8 and score['residual_ratio_with_margin'] <= .01
            if score['passes_main'] != passed:
                raise ValueError('Saved timing score decision mismatch')
        if not all(s['passes_main'] for s in record['scores']):
            raise ValueError('Saved timing record misses its accuracy target')
        durations[record['band'],record['pair'],record['kind']] = duration
    ratios_by_band = defaultdict(list)
    for pair in cold['pairs']:
        native = durations[pair['band'],pair['pair'],'glow']
        cut = durations[pair['band'],pair['pair'],'cut']
        if (not np.isclose(native,pair['native_cold_s'],rtol=1e-12,atol=1e-14)
                or not np.isclose(cut,pair['cut_cold_s'],rtol=1e-12,atol=1e-14)):
            raise ValueError('Saved paired timing duration mismatch')
        ratio = native/cut
        if not np.isclose(ratio,pair['native_over_cut'],rtol=1e-13):
            raise ValueError('Saved paired timing ratio mismatch')
        ratios_by_band[pair['band']].append(ratio)
    for spec in cold['band_summary']:
        ratios = ratios_by_band[spec['band']]
        if len(ratios) != 4 or not np.isclose(np.median(ratios),spec['paired_ratio_median'],rtol=1e-13):
            raise ValueError('Saved paired timing ratio mismatch')
        medians[str(spec['band'])] = float(np.median(ratios))
    return {'status':'PASS','frequency_cases':len(rows),
            'lens_configurations':len({(r['y'],r['kappa_s']) for r in rows}),
            'method_cases':sum(len(r['methods']) for r in rows),
            'band_metrics_checked':checked,'point_pass_counts':point_passes,
            'whole_band_pass_counts':passes,'saved_paired_cold_medians':medians,
            'source_artifacts_verified':len(manifest['files']),
            'reference_uncertainty_status':'empirical, not certified',
            'new_integrations_or_timings':False}


if __name__ == '__main__':
    print(json.dumps(replay(),indent=2))
