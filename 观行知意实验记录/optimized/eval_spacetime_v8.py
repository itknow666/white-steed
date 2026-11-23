"""Evaluate the crop geometry, temporal sampling and input resolution grid on original images."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import torch

from eval_local_motion_v8 import predictions, read
from eval_phase_order_v8 import RAW
from eval_sensor_robustness_v8 import BASE, INIT
from eval_shared_raw_v8 import shared_raw
from package_domain_v8 import ROOT
from train_videomae import sha, write


GRID = {f'f{frames}-s{size}-{geometry}': dict(
    input_frames=frames, input_size=size, input_geometry=geometry,
) for frames, size, geometry in product((16, 24, 32), (192, 224), ('stretch', 'letterbox'))}
PRIOR = ROOT/'submission/v8-input-resolution-20261123'
CONTROLS = {'f16-s192-stretch': PRIOR/'crop192-control', 'f16-s224-stretch': PRIOR/'crop224-direct'}
ARCHITECTURE = 'videomae2_dual_v8'
RECIPE_KEYS = ('seed', 'batch', 'epochs', 'base_lr', 'initial_sha256',
               'offline_teacher_sha256', 'teacher_policy', 'teacher_temperature', 'teacher_loss_weight')


def evaluate(runs, out):
    manifest = read(RAW/'manifest.json')
    options = {key: value for key, value in GRID.items() if key not in CONTROLS}
    initial_hash = sha(INIT)
    previous_runs = ROOT/'runs/v8-input-resolution-20261123'
    first_control = read(previous_runs/'crop192-control/config.json')
    recipe = {key: first_control[key] for key in RECIPE_KEYS}
    assert all(read(previous_runs/'crop224-direct/config.json')[key] == value for key, value in recipe.items())
    for key, spec in options.items():
        config = read(runs/key/'config.json')
        size = spec['input_size']
        assert config['input_shape'] == [4, spec['input_frames'], size, size]
        assert config['input_geometry'] == spec['input_geometry'] and config['initial_sha256'] == initial_hash
        assert all(config[key] == value for key, value in recipe.items())
    shared_raw(runs, out, RAW, manifest, dict.fromkeys(options, ARCHITECTURE), BASE,
               input_options=options)
    source_rows = {r['file']: r for r in read(RAW/'provenance.json')['samples']}
    subjects = {key: row['subject'] for key, row in source_rows.items()}
    people = {subject: f'S{i}' for i, subject in enumerate(sorted(set(subjects.values())), 1)}
    labels = {r.get('sample_id', r['file']+'.npy'): (int(r['label']), people[subjects[r['file']]])
              for r in manifest['samples']}
    original_frames = {r.get('sample_id', r['file']+'.npy'): source_rows[r['file']]['published_frames']
                       for r in manifest['samples']}
    groups = {(low, high): [key for key, count in original_frames.items() if low <= count <= high]
              for low, high in ((1, 15), (16, 23), (24, 31), (32, 32))}
    assert len(labels) == 466 and len(people) == 3
    paths = {'initial': BASE}
    paths.update({k: CONTROLS[k] if k in CONTROLS else out/k for k in GRID})
    baseline = predictions(BASE/'validation.csv')
    results = {}
    for name, folder in paths.items():
        row = read(folder/'metrics.json')
        current = predictions(folder/'validation.csv')
        assert current.keys() == baseline.keys() == labels.keys()
        results[name] = dict(row, by_subject={}, by_published_frames={})
        for group in people.values():
            subset = [key for key, (_, person) in labels.items() if person == group]
            mask = Counter((baseline[key] == labels[key][0], current[key] == labels[key][0]) for key in subset)
            results[name]['by_subject'][group] = dict(
                samples=len(subset), correct=mask[True, True] + mask[False, True],
                corrected=mask[False, True], regressed=mask[True, False],
            )
        paired = results[name]['by_subject'].values()
        assert abs(row['accuracy'] - 100 * sum(x['correct'] for x in paired)/466) < 1e-9
        results[name].update(corrected=sum(x['corrected'] for x in paired),
                             regressed=sum(x['regressed'] for x in paired))
        for (low, high), subset in groups.items():
            results[name]['by_published_frames'][f'{low}-{high}'] = dict(
                samples=len(subset), correct=sum(current[key] == labels[key][0] for key in subset),
            )
    best = max(results, key=lambda name: (results[name]['accuracy'], results[name]['macro_f1']))
    write(runs/'raw-comparison.json', {'recommended': best, 'results': results, 'settings': GRID,
                                       'reused_controls': list(CONTROLS), 'recipe': recipe})
    lines = [
        '# 方向 A · 帧数、分辨率与裁剪几何',
        '',
        '按 16/24/32 帧、192/224 输入、直接缩放/保比例补边做全组合。'
        '十组新实验各训两轮，复用上一轮两组经过原图验证的 16 帧直接缩放对照。'
        '各组从同一 73.61% 检查点开始，固定两轮、batch 8、seed 42、基础学习率 6e-6 与非目标蒸馏配方。',
        '',
        '| 方案 | 正确数 | 准确率 | Macro F1 | 改对 / 改错 | S1 正确 | S2 正确 | S3 正确 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, row in results.items():
        counts = [x['correct'] for x in row['by_subject'].values()]
        people_cells = ' | '.join(map(str, counts))
        lines.append(f'| {name} | {sum(counts)} | {row["accuracy"]:.2f}% | {row["macro_f1"]:.2f}% | '
                     f'{row["corrected"]} / {row["regressed"]} | {people_cells} |')
    lines += [
        '',
        f'按准确率与 Macro F1 排序，首选组为 {best}。',
        'S1/S2/S3 分别含 149/186/131 段。改对和改错均相对原始最佳权重计算。',
        '所有新候选均完成 466 段原图评估、三段标准接口对齐检查和单段 Windows Python 联网拦截。',
        '',
        '## 按原始帧数分组',
        '',
        '| 方案 | 原始 ≤15 帧 | 原始 16–23 帧 | 原始 24–31 帧 | 原始 32 帧 |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, row in results.items():
        cells = ' | '.join(f'{group["correct"]}/{group["samples"]}' for group in row['by_published_frames'].values())
        lines.append(f'| {name} | {cells} |')
    (runs/'实验结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(best, results[best]['accuracy'], results[best]['macro_f1'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.runs.resolve(), args.output.resolve())
