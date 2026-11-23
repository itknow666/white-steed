"""Validate matched-budget infrared teacher routing on 466 original clips."""
import argparse
from collections import Counter
from pathlib import Path
from shutil import copy2

import torch

from eval_local_motion_v8 import predictions, read
from eval_phase_order_v8 import RAW, compare
from eval_sensor_robustness_v8 import BASE
from eval_shared_raw_v8 import shared_raw
from package_domain_v8 import ROOT
from train_videomae import write


ARMS = dict.fromkeys(('ir-uniform', 'ir-class-random', 'ir-routed'), 'videomae2_dual_v8')
CONTROL = 'original-kd-control'
COLUMNS = ('原蒸馏续训', '均匀补充 IR', '逐类随机补充 IR', '按分差挑选 IR')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, out = args.runs.resolve(), args.output.resolve()
    manifest = read(RAW/'manifest.json')
    route = read(runs/'teacher-routing.json')
    previous = 'v8-input-resolution-20261123/crop192-control'
    ref = read(ROOT/'runs'/previous/'config.json')
    for name in ARMS:
        config = read(runs/name/'config.json')
        keys = ('input_shape', 'initial_sha256', 'seed', 'batch', 'base_lr', 'epochs',
                'teacher_policy', 'teacher_loss_weight', 'teacher_temperature', 'initial_trainable')
        assert all(config[key] == ref[key] for key in keys)
        assert config['offline_teacher_sha256'] == route['caches'][name]['sha256']
    original_control = ROOT/'submission'/previous
    control_dir = out/CONTROL
    control_dir.mkdir(parents=True, exist_ok=True)
    for file in ('metrics.json', 'validation.csv'):
        copy2(original_control/file, control_dir/file)

    shared_raw(runs, out, RAW, manifest, ARMS, BASE)
    compare(runs, out, manifest, arms={CONTROL: ARMS['ir-routed'], **ARMS}, columns=COLUMNS,
            title='红外教师等预算路由', baseline=BASE)

    sources = {row['file']: row['subject'] for row in read(RAW/'provenance.json')['samples']}
    names = {subject: f'S{i}' for i, subject in enumerate(sorted(set(sources.values())), 1)}
    labels = {r.get('sample_id', r['file']+'.npy'): (int(r['label']), names[sources[r['file']]])
              for r in manifest['samples']}
    control = predictions(control_dir/'validation.csv')
    assert len(labels) == 466 and labels.keys() == control.keys()
    rows = {}
    for name in ARMS:
        chosen = predictions(out/name/'validation.csv')
        assert chosen.keys() == labels.keys()
        by_subject = {}
        for subject in names.values():
            ids = [key for key, (_, person) in labels.items() if person == subject]
            cases = Counter((control[key] == labels[key][0], chosen[key] == labels[key][0]) for key in ids)
            by_subject[subject] = {
                'samples': len(ids),
                'control_correct': cases[True, True] + cases[True, False],
                'candidate_correct': cases[True, True] + cases[False, True],
                'corrected': cases[False, True], 'regressed': cases[True, False],
            }
        rows[name] = by_subject
    write(runs/'paired-subjects.json', rows)
    with (runs/'实验结果.md').open('a', encoding='utf-8') as report:
        report.write('\n## 相对原蒸馏续训对照\n\n| 组别 | 人物 | 样本数 | 对照正确 | 候选正确 | 改对 | 改错 |\n')
        report.write('|---|---|---:|---:|---:|---:|---:|\n')
        for name, groups in rows.items():
            for subject, counts in groups.items():
                report.write(f'| {name} | {subject} | ' + ' | '.join(map(str, counts.values())) + ' |\n')


if __name__ == '__main__':
    main()
