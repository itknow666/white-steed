"""Verify the local-alignment ablations against the published V8 single model."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from package_domain_v8 import ROOT, package, validate
from train_videomae import save, sha, write


BASE = ROOT/'submission/v8-paper-anchor43-single-20261122'
ARMS = ('matched-control', 'uniform', 'motion', 'motion-stratified')
MIDPOINTS = {'midpoint-control': 'matched-control', 'midpoint-motion': 'motion-stratified'}
INITIAL_WEIGHTS = ROOT/'runs/v8-paper-sprint-20261122/anchor43-half/best_ema.pt'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def predictions(path):
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return {row['sample_id']: int(row['predicted_class']) for row in rows}


def midpoint_checkpoints(runs):
    initial = torch.load(INITIAL_WEIGHTS, weights_only=True, map_location='cpu')
    for name, source in MIDPOINTS.items():
        folder = runs/name
        if (folder/'summary.json').exists():
            continue
        folder.mkdir(exist_ok=True)
        trained_path = runs/source/'best_ema.pt'
        trained = torch.load(trained_path, weights_only=True, map_location='cpu')
        if initial.keys() != trained.keys():
            raise ValueError('Midpoint checkpoint architectures differ')
        state = {key: torch.lerp(tensor, trained[key], .5) if tensor.is_floating_point() else tensor
                 for key, tensor in initial.items()}
        save(folder/'best_ema.pt', state)
        write(folder/'config.json', {'training_run': source, 'mix': .5, 'initial_sha256': sha(INITIAL_WEIGHTS),
                                     'trained_sha256': sha(trained_path)})
        write(folder/'summary.json', {'mode': name, 'source': source})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, out = args.runs.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(ROOT/'data/raw/validation/manifest.json')
    labels = {row.get('sample_id', row['file']+'.npy'): int(row['label']) for row in manifest['samples']}
    initial = predictions(BASE/'validation.csv')
    if initial.keys() != labels.keys():
        raise ValueError('Initial CSV and official validation differ')
    initial_misses = Counter((labels[key], initial[key]) for key in labels if labels[key] != initial[key])
    major_errors = [pair for pair, _ in initial_misses.most_common(5)]
    results = {'initial': read(BASE/'metrics.json')}
    for name in (*ARMS, *MIDPOINTS):
        dest = out/name
        result_path = dest/'metrics.json'
        if result_path.is_file():
            results[name] = read(result_path)
            continue
        if name in MIDPOINTS and not (runs/name/'summary.json').is_file():
            midpoint_checkpoints(runs)
        if not (runs/name/'summary.json').is_file():
            raise FileNotFoundError(runs/name/'summary.json')
        if not dest.exists():
            with TemporaryDirectory(dir=out) as temp:
                staged = Path(temp)/name
                package(runs, staged, {'mode': name, 'fusion': False, 'architecture': 'videomae2_dual_v8'})
                staged.rename(dest)
        raw, probe = validate(dest)
        current = predictions(dest/'validation.csv')
        if current.keys() != labels.keys():
            raise ValueError(f'{name} CSV and official validation differ')
        misses = Counter((labels[key], current[key]) for key in labels if labels[key] != current[key])
        corrected = sum(current[key] == labels[key] != initial[key] for key in labels)
        lost = sum(initial[key] == labels[key] != current[key] for key in labels)
        result = dict(raw, mode=name, parameters=probe['parameters'], corrected=corrected, regressed=lost,
                      scores_path='validation.csv', weights_sha256=sha(dest/'model.pt'),
                      code_sha256=sha(dest/'code.zip'), linux_verified=False,
                      error_counts=[misses[pair] for pair in major_errors])
        write(result_path, result)
        results[name] = result
    best = max(results, key=lambda name: (results[name]['accuracy'], results[name]['macro_f1']))
    write(runs/'raw-comparison.json', {'results': results, 'recommended': best})
    lines = ['# 方向 A 局部跨模态对齐', '',
             '从同一份 72.96% 轻量版权重出发，每组训练两轮。主对照与各对齐组共享训练前向路径。', '',
             '| 方案 | 原图准确率 | Macro F1 | 较起点改对 | 较起点改错 |', '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        lines.append(f"| {name} | {result['accuracy']:.2f}% | {result['macro_f1']:.2f}% | "
                     f"{result.get('corrected', 0)} | {result.get('regressed', 0)} |")
    lines += ['', f'基于公开验证的准确率与 Macro F1，推荐 `{best}`。各单模型推理参数量均为 24,767,544，含人物检测。',
              '两组 midpoint 各取起点与所列续训权重的一半，检验权重插值对普通续训和分段运动对齐的影响。',
              '', '## 起点最常见的五种误判', '',
              '| 真实类别 | 被预测为 | 起点 | 普通续训 | 随机选块 | 运动选块 | 分段运动选块 | 普通续训插值 | 分段运动插值 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for index, pair in enumerate(major_errors):
        y, p = pair
        counts = [initial_misses[pair], *(results[name]['error_counts'][index] for name in (*ARMS, *MIDPOINTS))]
        lines.append(f"| {manifest['classes'][y]} | {manifest['classes'][p]} | " + ' | '.join(map(str, counts)) + ' |')
    lines += ['', '各组代码 ZIP 通过 Windows Python 连接拦截下的单样本检查，并生成 466 条公开验证预测。',
              '对齐损失仅在训练时计算。报告使用公开验证集选型。']
    reference_path = runs/'control/summary.json'
    if reference_path.is_file():
        reference = read(reference_path)
        lines += ['', f"普通续训旧入口的缓存验证为 {reference['accuracy']:.2f}% / {reference['macro_f1']:.2f}%。"]
    (runs/'实验结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'recommended': best, 'results': results}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
