"""Probe frame-order sensitivity and compare phase heads on the full validation set."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from eval_local_motion_v8 import BASE, predictions, read
from package_domain_v8 import ROOT, package, validate
from train_videomae import metrics, save, sha, write


RAW = ROOT/'data/raw/validation'
ARMS = {'control': 'videomae2_dual_v8', 'phase': 'videomae2_phase_v8',
        'phase-order': 'videomae2_phase_v8', 'phase-head-only': 'videomae2_phase_v8',
        'phase-head-half': 'videomae2_phase_v8', 'pace-jitter': 'videomae2_dual_v8'}
PHASE_COLUMNS = ('普通续训', '阶段差分', '阶段差分加顺序损失', '仅训练阶段头', '阶段头分数减半', '节奏扰动')


def probe_order(folder, manifest):
    code = folder/'code'
    sys.path.insert(0, str(code))
    from inference import Model

    config = read(code/'config.json')
    config.update(weights_path=str(folder/'model.pt'), package_dir=str(code))
    model = Model(config, 'cuda')
    orders = {'normal': np.arange(16),
              'reverse_two_frame_groups': np.arange(16).reshape(8, 2)[::-1].ravel(),
              'swap_middle_four_frame_groups': np.arange(16).reshape(4, 4)[[0, 2, 1, 3]].ravel()}
    scores = {name: [] for name in orders}
    rows = manifest['samples']
    labels = np.array([int(row['label']) for row in rows])
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            prepared = model.prepare(RAW/row['file'])
            if prepared.shape[:2] != (4, 16):
                raise ValueError(f'Expected four channels and 16 frames, got {prepared.shape}')
            for name, order in orders.items():
                video = np.ascontiguousarray(prepared[:, order])
                scores[name].append(model._scores(model.model, video, False).cpu().numpy())
            if index % 100 == 0:
                print(f'Order probe {index}/{len(rows)}', flush=True)
    normal = np.array(scores['normal']).argmax(1)
    result = {}
    for name, values in scores.items():
        probabilities = np.array(values)
        chosen = probabilities.argmax(1)
        summary = metrics(labels, probabilities)
        summary.pop('confusion')
        result[name] = dict(summary, changed=int(np.sum(normal != chosen)), correct=int(np.sum(labels == chosen)))
    return {'samples': len(rows), 'results': result}


def compare(runs, out, manifest, *, arms=ARMS, columns=PHASE_COLUMNS, title='时序适配', training_sources=(), report=True):
    out.mkdir(parents=True, exist_ok=True)
    labels = {row.get('sample_id', row['file']+'.npy'): int(row['label']) for row in manifest['samples']}
    initial = predictions(BASE/'validation.csv')
    if initial.keys() != labels.keys():
        raise ValueError('Starting model CSV and validation manifest differ')
    errors = Counter((labels[key], initial[key]) for key in labels if labels[key] != initial[key])
    pairs = [pair for pair, _ in errors.most_common(5)]
    results = {'initial': read(BASE/'metrics.json')}
    for name, architecture in arms.items():
        folder = out/name
        if (folder/'metrics.json').is_file():
            results[name] = read(folder/'metrics.json')
            continue
        if name == 'phase-head-half' and not (runs/name/'summary.json').is_file():
            source = runs/'phase-head-only/best_ema.pt'
            checkpoint = torch.load(source, weights_only=True, map_location='cpu')
            checkpoint['phase_head.weight'].mul_(.5)
            (runs/name).mkdir(exist_ok=True)
            save(runs/name/'best_ema.pt', checkpoint)
            write(runs/name/'config.json', {'phase_head_multiplier': .5, 'source_sha256': sha(source)})
            write(runs/name/'summary.json', {'mode': name, 'source': 'phase-head-only'})
        if not (runs/name/'summary.json').is_file():
            raise FileNotFoundError(runs/name/'summary.json')
        if not folder.exists():
            with TemporaryDirectory(dir=out) as temp:
                staged = Path(temp)/name
                package(runs, staged, {'mode': name, 'fusion': False, 'architecture': architecture,
                                       'training_sources': training_sources})
                staged.rename(folder)
        raw, probe = validate(folder)
        current = predictions(folder/'validation.csv')
        if current.keys() != labels.keys():
            raise ValueError(f'Prediction IDs differ in {name}')
        mistakes = Counter((labels[key], current[key]) for key in labels if current[key] != labels[key])
        row = dict(raw, mode=name, parameters=probe['parameters'],
                   corrected=sum(current[k] == labels[k] != initial[k] for k in labels),
                   regressed=sum(initial[k] == labels[k] != current[k] for k in labels),
                   error_counts=[mistakes[pair] for pair in pairs],
                   weights_sha256=sha(folder/'model.pt'), code_sha256=sha(folder/'code.zip'), linux_verified=False)
        write(folder/'metrics.json', row)
        results[name] = row
    if not report:
        return
    best = max(results, key=lambda key: (results[key]['accuracy'], results[key]['macro_f1']))
    write(runs/'raw-comparison.json', {'results': results, 'recommended': best})
    lines = [f'# 方向 A · {title}', '', '| 方案 | 原图准确率 | Macro F1 | 改对 | 改错 | 推理参数 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in results.items():
        lines.append(f"| {name} | {row['accuracy']:.2f}% | {row['macro_f1']:.2f}% | "
                     f"{row.get('corrected', 0)} | {row.get('regressed', 0)} | {row['parameters']:,} |")
    lines += ['', f'按准确率与 Macro F1 选型，建议使用 `{best}`。', '',
              '| 真实类别 | 被预测为 | 起点 | ' + ' | '.join(columns) + ' |',
              '|---|---|' + '---:|' * (1 + len(columns))]
    for index, pair in enumerate(pairs):
        y, p = pair
        counts = [errors[pair], *(results[name]['error_counts'][index] for name in arms)]
        lines.append(f"| {manifest['classes'][y]} | {manifest['classes'][p]} | " + ' | '.join(map(str, counts)) + ' |')
    if (runs/'order-probe.json').is_file():
        probe = read(runs/'order-probe.json')
        lines += ['', '## 起点模型的顺序诊断', '',
                  '在原图预处理后，仅重排采样帧，沿用原标签衡量敏感性。', '',
                  '| 顺序 | 正确数 | 准确率 | 较正常顺序改变预测的片段数 |', '|---|---:|---:|---:|']
        for name, row in probe['results'].items():
            lines.append(f"| {name} | {row['correct']} | {row['accuracy']:.2f}% | {row['changed']} |")
    lines += ['', '各候选均完成 466 段 Windows 原图预测及单样本 Python 网络连接拦截检查。']
    (runs/'实验结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'recommended': best, 'results': results}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = read(RAW/'manifest.json')
    if args.probe_only:
        result = probe_order(BASE, manifest)
        write(args.runs/'order-probe.json', result)
        print(json.dumps(result), flush=True)
    else:
        if args.output is None:
            parser.error('--output is required for model comparison')
        compare(args.runs, args.output, manifest)


if __name__ == '__main__':
    main()
