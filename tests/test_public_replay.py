"""Public replay must work outside the repository working directory."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLD_PATH = 'data/timing/cold_confirmation.json'


@pytest.fixture(scope='module')
def replay_saved():
    return runpy.run_path(str(ROOT/'examples/replay_paper.py'))['replay']


@pytest.fixture
def cold_replay_fixture(tmp_path):
    """Copy real archived evidence; never mutate or link the source files."""
    fixture = tmp_path/'repository'
    fixture.mkdir()
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        target = fixture/item['public_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/item['public_path'], target)
    shutil.copy2(ROOT/'SOURCE_MANIFEST.json', fixture/'SOURCE_MANIFEST.json')
    return fixture


def write_cold_fixture(fixture, cold):
    """Keep fixture public hashes valid so rejection exercises semantic checks."""
    target = fixture/COLD_PATH
    target.write_text(json.dumps(cold, indent=2), encoding='utf-8')
    manifest_path = fixture/'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for item in manifest['files']:
        if item['public_path'] == COLD_PATH:
            item['public_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    assert len(manifest['files']) == 60
    for item in manifest['files']:
        assert hashlib.sha256((fixture/item['public_path']).read_bytes()).hexdigest() == item['public_sha256']


def test_paper_replay_checks_saved_values_from_another_working_directory(tmp_path):
    script = ROOT/'examples/replay_paper.py'
    assert script.is_file(), 'Public saved-value replay entry point is missing'
    run = subprocess.run([sys.executable, '-B', str(script)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout+run.stderr
    result = json.loads(run.stdout)
    assert result['status'] == 'PASS'
    assert result['frequency_cases'] == 5214
    assert result['lens_configurations'] == 30
    assert result['method_cases'] == 41712
    assert result['band_metrics_checked'] == 48
    assert result['whole_band_pass_counts']['cut_controlled'] == [6, 6]
    assert set(result['saved_paired_cold_medians']) == {'1', '2', '3', '4'}
    assert result['saved_paired_cold_medians'] == pytest.approx({
        '1': 3.658711592776074, '2': 6.1499502023667,
        '3': 3.928383353960659, '4': 6.34189541952843,
    }, rel=1e-13)
    assert result['source_artifacts_verified'] == 60
    assert result['new_integrations_or_timings'] is False


def test_public_manifest_cannot_read_outside_the_repository(tmp_path):
    replay = runpy.run_path(str(ROOT/'examples/replay_paper.py'))['replay']
    fixture = tmp_path/'repository'
    fixture.mkdir()
    outside = tmp_path/'outside.txt'
    outside.write_bytes(b'not a repository artifact')
    manifest = {'files':[{'public_path':'../outside.txt',
                         'public_sha256':hashlib.sha256(outside.read_bytes()).hexdigest(),
                         'transformation':'byte-identical copy'}]}
    (fixture/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='Unsafe public path'):
        replay(fixture)


@pytest.mark.parametrize('change', [
    'record_prepare_s', 'record_first_s', 'native_cold_s',
    'cut_cold_s', 'native_over_cut',
])
def test_cold_pairs_are_reconstructed_from_records(cold_replay_fixture, replay_saved, change):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    if change.startswith('record_'):
        record = cold['records'][0]
        record[change.removeprefix('record_')] *= 10
        record['cold_s'] = record['prepare_s'] + record['first_s']
    else:
        # This pair has band 1's maximum ratio: increasing it leaves the median unchanged.
        cold['pairs'][0][change] *= .5 if change == 'cut_cold_s' else 2
    write_cold_fixture(cold_replay_fixture, cold)
    with pytest.raises(ValueError, match='Saved paired timing (duration|ratio) mismatch'):
        replay_saved(cold_replay_fixture)


@pytest.mark.parametrize('collection', ['records', 'pairs'])
@pytest.mark.parametrize('change', ['missing', 'extra_duplicate', 'replace_duplicate', 'unknown_pair'])
def test_cold_replay_requires_unique_complete_identities(cold_replay_fixture, replay_saved,
                                                        collection, change):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    entries = cold[collection]
    if change == 'missing':
        entries.pop()
    elif change == 'extra_duplicate':
        entries.append(entries[0].copy())
    elif change == 'replace_duplicate':
        entries[1] = entries[0].copy()
    else:
        entries[0]['pair'] = 4
    write_cold_fixture(cold_replay_fixture, cold)
    with pytest.raises(ValueError, match='Incomplete saved timing records|Saved timing (record|pair) identity mismatch'):
        replay_saved(cold_replay_fixture)


@pytest.mark.parametrize('field,replacement', [('band', 5), ('pair', 4)])
def test_cold_replay_rejects_consistently_remapped_identifiers(cold_replay_fixture,
                                                             replay_saved, field, replacement):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    for collection in ['records', 'pairs']:
        for entry in cold[collection]:
            if entry['band'] == 1 and (field == 'band' or entry['pair'] == 0):
                entry[field] = replacement
    if field == 'band':
        cold['band_summary'][0]['band'] = replacement
    write_cold_fixture(cold_replay_fixture, cold)
    with pytest.raises(ValueError, match='Saved timing (record|pair) identity mismatch'):
        replay_saved(cold_replay_fixture)


def test_cold_replay_rejects_unknown_record_kind(cold_replay_fixture, replay_saved):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    cold['records'][0]['kind'] = 'unknown'
    write_cold_fixture(cold_replay_fixture, cold)
    with pytest.raises(ValueError, match='Saved timing record identity mismatch'):
        replay_saved(cold_replay_fixture)


def test_cold_replay_joins_by_keys_not_archive_order(cold_replay_fixture, replay_saved):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    cold['records'].reverse()
    cold['pairs'] = cold['pairs'][7:] + cold['pairs'][:7]
    cold['band_summary'].reverse()
    write_cold_fixture(cold_replay_fixture, cold)
    result = replay_saved(cold_replay_fixture)
    assert result['status'] == 'PASS'
    assert result['saved_paired_cold_medians'] == {
        str(spec['band']): spec['paired_ratio_median'] for spec in cold['band_summary']
    }


@pytest.mark.parametrize('change', [
    'empty', 'missing', 'extra_duplicate', 'replace_duplicate', 'unknown_band',
])
def test_cold_replay_requires_exactly_four_unique_band_summaries(cold_replay_fixture,
                                                               replay_saved, change):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    summaries = cold['band_summary']
    if change == 'empty':
        summaries.clear()
    elif change == 'missing':
        summaries.pop()
    elif change == 'extra_duplicate':
        summaries.append(summaries[0].copy())
    elif change == 'replace_duplicate':
        summaries[1] = summaries[0].copy()
    else:
        summaries[0]['band'] = 5
    write_cold_fixture(cold_replay_fixture, cold)
    with pytest.raises(ValueError, match='Saved timing band summary identity mismatch'):
        replay_saved(cold_replay_fixture)


@pytest.mark.parametrize('score_index', [0, 1], ids=['first', 'cached'])
@pytest.mark.parametrize('metric,value', [
    ('max_with_margin', 2e-8), ('residual_ratio_with_margin', .02),
])
def test_cold_replay_checks_numeric_main_accuracy_decisions(cold_replay_fixture,
                                                          replay_saved, score_index, metric, value):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    cold['records'][-1]['scores'][score_index][metric] = value
    write_cold_fixture(cold_replay_fixture, cold)
    with pytest.raises(ValueError, match='Saved timing score decision mismatch'):
        replay_saved(cold_replay_fixture)


def test_cold_replay_rejects_false_main_accuracy_flag(cold_replay_fixture, replay_saved):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    cold['records'][0]['scores'][0]['passes_main'] = False
    write_cold_fixture(cold_replay_fixture, cold)
    with pytest.raises(ValueError, match='Saved timing (score decision mismatch|record misses its accuracy target)'):
        replay_saved(cold_replay_fixture)


def test_cold_replay_keeps_inclusive_main_accuracy_thresholds(cold_replay_fixture, replay_saved):
    cold = json.loads((cold_replay_fixture/COLD_PATH).read_text(encoding='utf-8'))
    score = cold['records'][0]['scores'][0]
    score['max_with_margin'] = 1e-8
    score['residual_ratio_with_margin'] = .01
    write_cold_fixture(cold_replay_fixture, cold)
    assert replay_saved(cold_replay_fixture)['status'] == 'PASS'
