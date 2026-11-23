"""Evaluate late local adapters on the official original-image validation set."""
import argparse
from collections import Counter
from pathlib import Path
from shutil import copy2

import torch

from eval_local_motion_v8 import predictions, read
from eval_phase_order_v8 import RAW, compare
from eval_sensor_robustness_v8 import BASE, INIT
from eval_shared_raw_v8 import shared_raw
from package_domain_v8 import ROOT
from train_videomae import sha, write
from videomae_v8 import LATE_ADAPTER_KERNELS


PILOT = [f'{scope}-{variant}' for scope in ('frozen', 'full') for variant in LATE_ADAPTER_KERNELS]
PRIOR = 'v8-input-resolution-20261123/crop192-control'
CONTROL = 'original-kd-control'
PLAIN = 'plain-control'


def architecture(name):
    return 'videomae2_dual_v8' if name == PLAIN else f'videomae2_{name.split("-")[1]}_v8'


def paired(runs, out, manifest, arms, control_name):
    source = {row['file']: row['subject'] for row in read(RAW/'provenance.json')['samples']}
    people = {subject: f'S{i}' for i, subject in enumerate(sorted(set(source.values())), 1)}
    truth = {row.get('sample_id', row['file']+'.npy'): (int(row['label']), people[source[row['file']]])
             for row in manifest['samples']}
    assert len(truth) == 466 and len(people) == 3
    cache = {name: predictions(out/name/'validation.csv') for name in set(arms) | {control_name}}
    assert all(truth.keys() == values.keys() for values in cache.values())
    rows = {}
    for name in arms:
        reference = name.split('-')[0]+'-latepoint'
        if reference == name or reference not in cache:
            reference = control_name
        chosen = cache[name]
        control = cache[reference]
        by_subject = {}
        for person in people.values():
            keys = [key for key, (_, subject) in truth.items() if subject == person]
            pair = Counter((control[k] == truth[k][0], chosen[k] == truth[k][0]) for k in keys)
            by_subject[person] = {'samples': len(keys), 'control_correct': pair[True,True] + pair[True,False],
                                  'candidate_correct': pair[True,True] + pair[False,True],
                                  'corrected': pair[False,True], 'regressed': pair[True,False]}
        rows[name] = {'control': reference, 'subjects': by_subject}
    write(runs/'paired-subjects.json', rows)
    with (runs/'实验结果.md').open('a', encoding='utf-8') as output:
        output.write('\n## 按人物配对比较\n\n')
        output.write('| 候选 | 配对对照 | 人物 | 样本数 | 对照正确 | 候选正确 | 改对 | 改错 |\n')
        output.write('|---|---|---|---:|---:|---:|---:|---:|\n')
        for name, entry in rows.items():
            for person, values in entry['subjects'].items():
                cells = ' | '.join(map(str, values.values()))
                output.write(f'| {name} | {entry["control"]} | {person} | {cells} |\n')


def evaluate(runs, out, seed, selected, frozen_lr, plain_reference):
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(RAW/'manifest.json')
    known = read(ROOT/'runs'/PRIOR/'config.json')
    initial_hash = sha(INIT)
    arms = dict.fromkeys(selected)
    for name in arms:
        if name not in (*PILOT, PLAIN):
            raise ValueError(f'Unrecognized adapter trial {name}')
        cfg = read(runs/name/'config.json')
        fields = ('batch', 'epochs', 'cache_sha256', 'offline_teacher_sha256', 'teacher_policy',
                  'teacher_temperature', 'teacher_loss_weight', 'input_shape')
        assert all(cfg[field] == known[field] for field in fields)
        assert cfg['seed'] == seed and cfg['initial_sha256'] == initial_hash
        assert cfg['input_geometry'] == known.get('input_geometry', 'stretch')
        assert cfg['base_lr'] == (frozen_lr if name.startswith('frozen-') else 6e-6)
        if name != PLAIN:
            variant = name.split('-')[1]
            assert cfg['domain_variant'] == variant and cfg['late_adapter_blocks'] == [9, 11]
            assert cfg['late_adapter_kernel'] == list(LATE_ADAPTER_KERNELS[variant])
            assert cfg['late_adapters_only'] is name.startswith('frozen-')
        arms[name] = architecture(name)
    control = PLAIN
    reused = seed == 42 or plain_reference is not None
    if reused:
        control = CONTROL if seed == 42 else PLAIN
        original = plain_reference or ROOT/'submission'/PRIOR
        target = out/control
        target.mkdir(exist_ok=True)
        for filename in ('metrics.json', 'validation.csv'):
            copy2(original/filename, target/filename)
    elif PLAIN not in arms:
        raise ValueError('A new seed requires its own no-adapter control')
    shared_raw(runs, out, RAW, manifest, arms, BASE)
    compared = {control: 'videomae2_dual_v8', **arms} if reused else arms
    compare(runs, out, manifest, arms=compared, columns=tuple(compared),
            title=f'VideoMAE 末端卷积适配器（seed {seed}）', baseline=BASE)
    paired(runs, out, manifest, arms, control)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arm', action='append')
    parser.add_argument('--frozen-lr', type=float, default=2.4e-5)
    parser.add_argument('--plain-reference', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.runs.resolve(), args.output.resolve(), args.seed, args.arm or PILOT,
             args.frozen_lr, args.plain_reference)
