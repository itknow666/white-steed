"""Validate direct-from-original 224 crops against the existing 192 path."""
import argparse
from collections import Counter
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import torch

from eval_phase_order_v8 import RAW, compare
from eval_sensor_robustness_v8 import BASE, INIT
from eval_local_motion_v8 import predictions, read
from package_domain_v8 import package
from train_videomae import sha, write


SIZES = {'crop192-control': 192, 'crop224-direct': 224, 'crop224-no-training': 224}
ARCHITECTURE = 'videomae2_dual_v8'


def paired_subjects(runs, out, manifest):
    provenance = read(RAW/'provenance.json')
    subjects = {row['file']: row['subject'] for row in provenance['samples']}
    truth = {row.get('sample_id', row['file']+'.npy'): (int(row['label']), subjects[row['file']])
             for row in manifest['samples']}
    assert len(truth) == 466 and len(set(subjects.values())) == 3
    candidates = {'initial': predictions(BASE/'validation.csv')}
    candidates.update({arm: predictions(out/arm/'validation.csv') for arm in SIZES})
    assert all(p.keys() == truth.keys() for p in candidates.values())
    comparisons = [('initial', 'crop224-no-training'), ('initial', 'crop192-control'),
                   ('initial', 'crop224-direct'), ('crop192-control', 'crop224-direct')]
    rows = {}
    for reference, candidate in comparisons:
        categories = {}
        for index, subject in enumerate(sorted(set(subjects.values())), 1):
            ids = [key for key, (_, person) in truth.items() if person == subject]
            paired = Counter((candidates[reference][key] == truth[key][0],
                              candidates[candidate][key] == truth[key][0]) for key in ids)
            categories[f'S{index}'] = {'samples': len(ids), 'reference_correct': paired[(True, True)] + paired[(True, False)],
                                       'candidate_correct': paired[(True, True)] + paired[(False, True)],
                                       'corrected': paired[(False, True)], 'regressed': paired[(True, False)]}
        rows[f'{reference} -> {candidate}'] = categories
    write(runs/'paired-subjects.json', rows)
    result = runs/'实验结果.md'
    with result.open('a', encoding='utf-8') as report:
        report.write('\n## 按人物逐段对照\n\n')
        report.write('| 比较 | 人物 | 样本数 | 对照正确 | 候选正确 | 改对 | 改错 |\n')
        report.write('|---|---|---:|---:|---:|---:|---:|\n')
        for comparison, people in rows.items():
            for subject, values in people.items():
                report.write(f'| {comparison} | {subject} | ' + ' | '.join(map(str, values.values())) + ' |\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--early', action='store_true', help='Evaluate the first two available packages while the 224 cache builds')
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, out = args.runs.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(RAW/'manifest.json')
    probe_run = runs/'crop224-no-training'
    probe_run.mkdir(exist_ok=True)
    write(probe_run/'summary.json', {'training_steps': 0, 'initial_sha256': sha(INIT), 'input_size': 224})
    sizes = {name: SIZES[name] for name in ('crop224-no-training', 'crop192-control')} if args.early else SIZES
    for name, size in sizes.items():
        folder = out/name
        if folder.exists():
            assert read(folder/'code/config.json')['size'] == size
            continue
        with TemporaryDirectory(dir=out) as temporary:
            temporary = Path(temporary)
            source = runs
            if name == 'crop224-no-training':
                source = temporary/'checkpoint'
                (source/name).mkdir(parents=True)
                copy2(INIT, source/name/'best_ema.pt')
                assert sha(source/name/'best_ema.pt') == sha(INIT)
            else:
                cfg = read(runs/name/'config.json')
                assert cfg['input_shape'] == [4, 16, size, size] and cfg['initial_sha256'] == sha(INIT)
            staged = temporary/name
            package(source, staged, {'mode': name, 'architecture': ARCHITECTURE,
                                    'fusion': False, 'input_size': size})
            staged.rename(folder)
    compare(runs, out, manifest, arms=dict.fromkeys(sizes, ARCHITECTURE), report=not args.early,
            columns=('192续训对照', '原图直采224续训', '原图直采224只换推理输入'),
            title='原图直采224输入', baseline=BASE)
    if not args.early:
        paired_subjects(runs, out, manifest)


if __name__ == '__main__':
    main()
