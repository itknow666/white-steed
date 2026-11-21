"""Rank completed direction-A sprint runs with fixed fusion, then verify raw images."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np

from package_domain_v8 import ROOT, R34, package, predicted, validate
from train_videomae import metrics, sha, write


RUN_NAMES = ('transfer43', 'transfer44', 'staged43', 'plain', 'temporal', 'gate', 'masked', 'distill',
             'soup43-v8', 'soup43-44', 'soup-three', 'anchor43-half')
BASE = ROOT/'runs/v8-domain-20261122/dual'
BASE_PACKAGE = ROOT/'submission/v8-domain-20261122'
ARCH = {'gate': 'videomae2_gate_v8', 'temporal': 'videomae2_temporal_v8'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(runs):
    ids, r34, labels = predicted(R34/'cached-single-predictions.npz')
    scores = {}
    sources = [(f'v7-source{seed}', ROOT/f'runs/videomae2_s-v7-seed{seed}-20261114') for seed in (43, 44)]
    references = [('v8-original', BASE), ('v7-source-soup', runs/'v7-source-soup')]
    for name, path in references + sources + [(n, runs/n) for n in RUN_NAMES]:
        if not (path/'summary.json').is_file():
            continue
        branch_ids, probs, targets = predicted(path/'best_predictions.npz')
        indices = {k: i for i, k in enumerate(branch_ids)}
        order = [indices[k] for k in ids]
        if not np.array_equal(labels, targets[order]):
            raise ValueError('Validation labels differ')
        one = metrics(labels, probs[order])
        two = metrics(labels, .75*probs[order] + .25*r34)
        for m in (one, two):
            m.pop('confusion')
        scores[name] = {'single': one, 'fusion': two, 'epochs': read(path/'summary.json')['best_epoch']}
    completed = [name for name in RUN_NAMES if name in scores]
    if not completed:
        raise RuntimeError('No full training arm completed')
    winner = max(completed, key=lambda name: (scores[name]['fusion']['accuracy'], scores[name]['fusion']['macro_f1']))
    write(runs/'comparison.json', {'scores': scores, 'selected_for_raw_validation': winner,
                                  'incomplete': [n for n in RUN_NAMES if n not in scores]})
    return winner, scores


def paired(output):
    def predictions(path):
        with path.open(newline='', encoding='utf-8') as f:
            return {r['sample_id']: int(r['predicted_class']) for r in csv.DictReader(f)}
    manifest = read(ROOT/'data/raw/validation/manifest.json')
    labels = {s.get('sample_id', s['file']+'.npy'): int(s['label']) for s in manifest['samples']}
    older = predictions(BASE_PACKAGE/'validation.csv')
    newer = predictions(output/'validation.csv')
    if newer.keys() != labels.keys() or older.keys() != labels.keys():
        raise ValueError('Raw validation sample IDs differ')
    corrected = sum(newer[k] == labels[k] != older[k] for k in labels)
    lost = sum(older[k] == labels[k] != newer[k] for k in labels)
    cache = read(ROOT/'cache/crop16-192-v3/manifest.json')
    by_subject = Counter()
    for row in cache['samples']:
        if row['split'] == 'validation':
            key = row['sample_id']
            label = labels[key]
            by_subject[row['subject']] += int(newer[key] == label) - int(older[key] == label)
    miss = Counter((labels[k], newer[k]) for k in labels if newer[k] != labels[k])
    classes = manifest['classes']
    return {'corrected': corrected, 'regressed': lost, 'net': corrected-lost,
            'subjects_improved': sum(delta > 0 for delta in by_subject.values()),
            'subjects_tied': sum(delta == 0 for delta in by_subject.values()),
            'subjects_regressed': sum(delta < 0 for delta in by_subject.values()),
            'most_common_errors': [{'true': classes[y], 'predicted': classes[yhat], 'count': count}
                                   for (y, yhat), count in miss.most_common(5)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs, out = args.runs.resolve(), args.output.resolve()
    deadline_file = runs/'soup-deadline.txt'
    if deadline_file.is_file():
        value = deadline_file.read_text(encoding='utf-8-sig').strip()
        # PowerShell's round-trip format emits seven fractional-second digits.
        deadline = datetime.fromisoformat(re.sub(r'(\.\d{6})\d+', r'\1', value))
        while not (runs/'soup-exit.txt').is_file() and datetime.now(timezone.utc) < deadline:
            time.sleep(4)
        while (runs/'staged43.log').is_file() and not (runs/'staged43-exit.txt').is_file() and datetime.now(timezone.utc) < deadline:
            time.sleep(4)
    winner, scores = summarize(runs)
    config = read(runs/winner/'config.json')
    architecture = ARCH.get(config.get('domain_variant', 'plain'), 'videomae2_dual_v8')
    package(runs, out, {'mode': winner, 'fusion': True, 'architecture': architecture,
                        'training_sources': ('soup_domain_v8.py', 'run_v8_paper_sprint.py')})
    raw, probe = validate(out)
    prev = read(BASE_PACKAGE/'metrics.json')['raw_validation']
    use_candidate = (raw['accuracy'],raw['macro_f1']) > (prev['accuracy'],prev['macro_f1'])
    changes = paired(out)
    result = {'mode': winner, 'raw_validation': raw, 'previous_raw_validation': prev,
              'pairs': changes, 'promote': use_candidate, 'parameters': probe['parameters'],
              'architecture': architecture, 'weights_sha256': sha(out/'model.pt'),
              'code_sha256': sha(out/'code.zip'), 'linux_verified': False}
    write(out/'metrics.json', result)
    lines = ['# 方向 A 半小时论文实验结果', '',
             '| 实验 | 单模型准确率 | 单模型 Macro F1 | 固定融合准确率 | 固定融合 Macro F1 |',
             '|---|---:|---:|---:|---:|']
    for name, data in scores.items():
        lines.append(f"| {name} | " + ' | '.join(f'{data[k][m]:.2f}%' for k in ('single','fusion') for m in ('accuracy','macro_f1'))+' |')
    lines += ['', f"最高缓存候选为 {winner}。466 段原图复验为 **{raw['accuracy']:.2f}% / {raw['macro_f1']:.2f}%**。",
              f"与原 V8 配对比较，改对 {changes['corrected']} 段，改错 {changes['regressed']} 段。",
              f"按人物分组，{changes['subjects_improved']} 人正确数增加，{changes['subjects_tied']} 人持平，{changes['subjects_regressed']} 人减少。",
              f"保留{'本轮候选' if use_candidate else '原 V8'}作为方向 A 推荐版本。推理参数量 {probe['parameters']:,}。",
              '', '从实际代码 ZIP 通过 Windows Python 网络拦截下的单样本加载检查。',
              '', 'v7-source43、v7-source44 和 v7-source-soup 为双路迁移之前的同源对照。v7-source-soup 对三个 V7 种子的原架构检查点取等权平均。',
              '', '训练组每组两轮，单训练随机种子。权重平均组直接复用训练好的检查点。公开验证集用于选型。',
              '', f"未完成组别：{', '.join(n for n in RUN_NAMES if n not in scores) or '无'}。"]
    lines += ['', '## 原图中仍最常见的误判', '', '| 真实类别 | 预测类别 | 段数 |', '|---|---|---:|']
    for error in changes['most_common_errors']:
        lines.append(f"| {error['true']} | {error['predicted']} | {error['count']} |")
    (runs/'实验结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
