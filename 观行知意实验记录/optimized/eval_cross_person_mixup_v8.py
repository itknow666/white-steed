"""Compare Mixup partner policies and their original-image predictions by person."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from eval_local_motion_v8 import predictions, read
from eval_phase_order_v8 import RAW
from eval_sensor_robustness_v8 import BASE, INIT
from eval_shared_raw_v8 import shared_raw
from package_domain_v8 import ROOT
from train_videomae import sha, write


ARMS = dict.fromkeys(('random', 'distinct', 'cross-person'), 'videomae2_dual_v8')
SEEDS = {42: Path('.'), 43: Path('seed43')}


def load_training(folder, seed):
    config = read(folder/'config.json')
    assert config['seed'] == seed and config['mixup_partners'] == folder.name
    assert config['initial_sha256'] == sha(INIT) and config['input_shape'] == [4, 16, 192, 192]
    assert config['mixup_probability'] == .5 and config['mixup_beta'] == .2
    assert config['teacher_policy'] == 'non-target' and config['teacher_loss_weight'] == .35
    history = [json.loads(line) for line in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(history) == 2 and history[-1]['steps'] == 500 and all(r['seen'] == 1995 for r in history)
    epochs = [row['mixup'] for row in history]
    counts = {key: sum(e[key] for e in epochs) for key in epochs[0]}
    return config, epochs, counts


def evaluate(runs, out):
    manifest = read(RAW/'manifest.json')
    subject_codes = {r['file']: r['subject'] for r in read(RAW/'provenance.json')['samples']}
    people = {s: f'S{i}' for i, s in enumerate(sorted(set(subject_codes.values())), 1)}
    truth = {}
    for row in manifest['samples']:
        key = row.get('sample_id', row['file']+'.npy')
        truth[key] = int(row['label']), people[subject_codes[row['file']]]
    assert len(truth) == 466 and len(people) == 3
    all_rows = {}
    for seed, relative in SEEDS.items():
        subset, packages = runs/relative, out/relative
        configs, schedules, train = {}, {}, {}
        for arm in ARMS:
            configs[arm], schedules[arm], train[arm] = load_training(subset/arm, seed)
        for arm in ARMS:
            assert all(configs[arm][key] == value for key, value in configs['random'].items()
                       if key != 'mixup_partners')
            assert all(all(a[key] == b[key] for key in ('batches', 'pairs','partner_weight'))
                       for a, b in zip(schedules[arm], schedules['random']))
        assert train['distinct']['self_pairs'] == train['cross-person']['self_pairs'] == 0
        assert train['cross-person']['cross_person'] >= train['distinct']['cross_person']
        shared_raw(subset, packages, RAW, manifest, ARMS, BASE)
        random_predictions = predictions(packages/'random'/'validation.csv')
        assert random_predictions.keys() == truth.keys()
        rows = {}
        for arm in ARMS:
            folder = packages/arm
            row = read(folder/'metrics.json')
            current = predictions(folder/'validation.csv')
            assert current.keys() == truth.keys()
            paired = {}
            for person in people.values():
                counts = Counter((random_predictions[k] == label, current[k] == label)
                                 for k, (label, s) in truth.items() if s == person)
                paired[person] = {'correct': counts[True,True]+counts[False,True],
                                  'corrected': counts[False,True], 'regressed': counts[True,False]}
            assert sum(counts['correct'] for counts in paired.values()) == round(row['accuracy'] * 466/100)
            rows[arm] = dict(row, training_mixup=train[arm], by_subject=paired)
        all_rows[str(seed)] = rows

    prior = read(ROOT/'submission/v8-input-resolution-20261123/crop192-control/metrics.json')
    candidates = [(seed, arm) for seed, rows in all_rows.items() for arm in rows]
    seed, best = max(candidates, key=lambda key:(all_rows[key[0]][key[1]]['accuracy'],all_rows[key[0]][key[1]]['macro_f1']))
    top = all_rows[seed][best]
    improved = (top['accuracy'],top['macro_f1']) > (prior['accuracy'],prior['macro_f1'])
    summary = {'results': all_rows, 'samples': 466, 'best_new': {'seed': int(seed),'arm': best,
                'improved_over_prior': improved}}
    write(runs/'raw-comparison.json', summary)
    lines = [
        '# 方向 A · 跨人物 Mixup 双种子验证',
        '',
        '同一份 73.61% 权重、原非目标蒸馏、16 帧和 192 输入。每组两轮，batch 8。'
        '同种子三组使用同一批序、Mixup 触发批次以及相同的混合系数。',
        '',
        '| 种子 | 配对 | 正确数 | 准确率 | Macro F1 | 跨人物比例 | 不同动作比例 | 自配对 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for seed, rows in all_rows.items():
        for name, row in rows.items():
            m = row['training_mixup']
            lines.append(f'| {seed} | {name} | {round(row["accuracy"]*466/100)} | {row["accuracy"]:.2f}% | '
                         f'{row["macro_f1"]:.2f}% | {m["cross_person"]/m["pairs"]:.2%} | '
                         f'{m["different_label"]/m["pairs"]:.2%} | {m["self_pairs"]} |')
    lines += ['', '## 相对同种子随机对照', '',
              '| 种子 | 配对 | S1 改对/改错 | S2 改对/改错 | S3 改对/改错 |',
              '|---|---|---:|---:|---:|']
    for seed, rows in all_rows.items():
        for name in ('distinct','cross-person'):
            values = rows[name]['by_subject']
            counts = ' | '.join(f'{v["corrected"]}/{v["regressed"]}' for v in values.values())
            lines.append(f'| {seed} | {name} | {counts} |')
    lines += ['', '所有新组均完成 466 段原图验证、三段标准 Model.predict 逐分数核对以及 Windows 单段 Python 联网拦截。']
    (runs/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary['best_new'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.runs.resolve(),args.output.resolve())
