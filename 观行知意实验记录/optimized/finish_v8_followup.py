"""Evaluate V8 follow-up arms against the unchanged V8 checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np

from package_domain_v8 import ROOT, R34, package, predicted, validate
from train_videomae import metrics, sha, write


ARMS = ('control', 'ir-jitter', 'adapter-only', 'class-weight')
BASELINE = ROOT/'runs/v8-domain-20261122/dual'
BASELINE_PACKAGE = ROOT/'submission/v8-domain-20261122'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs, output = args.runs.resolve(), args.output.resolve()
    ids, r34, labels = predicted(R34/'cached-single-predictions.npz')
    comparisons = {}
    for name, directory in [('v8-original', BASELINE), *[(a, runs/a) for a in ARMS]]:
        branch_ids, branch, targets = predicted(directory/'best_predictions.npz')
        index = {key: i for i, key in enumerate(branch_ids)}
        order = [index[key] for key in ids]
        if not np.array_equal(targets[order], labels):
            raise ValueError('Validation target mismatch')
        single = metrics(labels, branch[order])
        fusion = metrics(labels, .75 * branch[order] + .25 * r34)
        for result in (single, fusion):
            result.pop('confusion')
        comparisons[name] = {'single': single, 'fusion': fusion}
    winner = max(ARMS, key=lambda n: (comparisons[n]['fusion']['accuracy'], comparisons[n]['fusion']['macro_f1']))
    write(runs/'comparison.json', {'scores': comparisons, 'selected': winner})
    package(runs, output, {'mode': winner, 'fusion': True})
    raw, probe = validate(output)
    previous = json.loads((BASELINE_PACKAGE/'metrics.json').read_text(encoding='utf-8'))['raw_validation']
    promote = (raw['accuracy'], raw['macro_f1']) > (previous['accuracy'], previous['macro_f1'])
    result = {'mode': winner, 'raw_validation': raw, 'previous_raw_validation': previous, 'promote': promote,
              'parameters': probe['parameters'], 'weights_sha256': sha(output/'model.pt'),
              'code_sha256': sha(output/'code.zip'), 'linux_verified': False}
    write(output/'metrics.json', result)
    lines = ['# V8 方向 A 续训对照', '',
             '四组从 V8 dual 最佳权重出发，按人物划分的训练与验证集不变。各训一轮，基础学习率 1e-5，batch 8，seed 42。'
             '候选融合固定使用 VideoMAE 0.75、旧 R34 0.25。', '',
             '| 组别 | 单模型准确率 | 单模型 Macro F1 | 融合准确率 | 融合 Macro F1 |',
             '|---|---:|---:|---:|---:|']
    for name, modes in comparisons.items():
        row = [modes[k][metric] for k in ('single', 'fusion') for metric in ('accuracy', 'macro_f1')]
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}%" for v in row) + " |")
    lines += ['', f"原图复验候选为 {winner}，准确率 {raw['accuracy']:.2f}%，Macro F1 {raw['macro_f1']:.2f}%。",
              f"分支推荐权重：{'续训候选' if promote else '现有 V8'}。模型参数 {probe['parameters']:,}，复验覆盖 466 段。",
              '', 'Windows Python 网络拦截的单样本加载检查通过。指标源于同一公开验证集，用于选型。']
    (runs/'实验结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
