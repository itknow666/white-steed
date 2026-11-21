"""Compare V8 runs, export the selected model, and verify the real-image path."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch

from train_videomae import metrics, sha, write


ROOT = Path(__file__).resolve().parents[1]
R34 = ROOT/'runs/control16-v4-20261027'
V7 = ROOT/'submission/fusion-videomae2-r34-v7-20261114'
MODES = ('control', 'dual', 'lora', 'staged', 'ssl')


def predicted(path):
    with np.load(path) as data:
        ids = data['sample_ids'].tolist()
        if len(set(ids)) != 466:
            raise ValueError('Expected 466 unique validation IDs')
        return ids, data['probabilities'], data['labels']


def compare(root):
    ids, r34, labels = predicted(R34/'cached-single-predictions.npz')
    results = []
    for mode in MODES:
        folder = root/mode
        if not (folder/'summary.json').is_file():
            raise FileNotFoundError(folder/'summary.json')
        sample_ids, probs, targets = predicted(folder/'best_predictions.npz')
        positions = {key: i for i, key in enumerate(sample_ids)}
        reorder = [positions[key] for key in ids]
        if not np.array_equal(labels, targets[reorder]):
            raise ValueError('Validation labels differ')
        for f in (False, True):
            values = .75 * probs[reorder] + .25 * r34 if f else probs[reorder]
            result = metrics(labels, values)
            result.pop('confusion')
            results.append(dict(result, mode=mode, fusion=f))
    alternatives = [x for x in results if x['mode'] != 'control']
    winner = max(alternatives, key=lambda r: (r['accuracy'], r['macro_f1'], not r['fusion']))
    write(root/'comparison.json', {'comparison': results, 'selected_v8': winner})
    return winner


def package(run, package_dir, winner):
    package_dir.mkdir(parents=True)
    config = json.loads((V7/'code/config.json').read_text(encoding='utf-8'))
    arch = 'videomae2_lora_v8' if winner['mode'] == 'lora' else 'videomae2_dual_v8'
    config['components'][0]['architecture'] = arch
    state = torch.load(run/winner['mode']/'best_ema.pt', map_location='cpu', weights_only=True)
    if winner['fusion']:
        r34 = torch.load(R34/'best_ema.pt', map_location='cpu', weights_only=True)
        state = {f'{i}.{name}': tensor for i, branch in enumerate((state, r34))
                 for name, tensor in branch.items()}
    else:
        config['architecture'] = arch
        config.pop('components')
    torch.save(state, package_dir/'model.pt')
    additions = {'inference.py': ROOT/'optimized/inference_v7.py',
                 'videomae_v8.py': ROOT/'optimized/videomae_v8.py',
                 'optimized/train_domain_v8.py': ROOT/'optimized/train_domain_v8.py',
                 'optimized/train_videomae.py': ROOT/'optimized/train_videomae.py'}
    archive = package_dir/'code.zip'
    with zipfile.ZipFile(V7/'code.zip') as base, zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as out:
        for name in base.namelist():
            if name not in additions and name not in ('config.json', 'README.md'):
                out.writestr(name, base.read(name))
        for name, path in additions.items():
            out.write(path, name)
        out.writestr('config.json', json.dumps(config, indent=2))
        out.writestr('README.md', 'V8 domain adaptation. Run predict.py with matching model.pt. '
                     'Training: optimized/train_domain_v8.py with synchronized cache and V7 VideoMAE weight.\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) <= 256
        assert sum(f.file_size for f in z.infolist()) <= 64*1024**2 and archive.stat().st_size <= 20*1024**2
        z.extractall(package_dir/'code')
    model = package_dir/'model.pt'
    shutil.copy2(model, package_dir/'code/model.pt')
    return config


def validate(dest):
    data = ROOT/'data/raw/validation'
    manifest = data/'manifest.json'
    raw = json.loads(manifest.read_text(encoding='utf-8'))
    code = dest/'code'
    subprocess.run([sys.executable, str(ROOT/'tools/offline_probe.py'), '--package', str(dest/'code.zip'),
                    '--weights', str(dest/'model.pt'), '--sample', str(data/raw['samples'][0]['file']),
                    '--output', str(dest/'offline-check.json')], check=True)
    subprocess.run([sys.executable, str(code/'predict.py'), '--dataset', str(data), '--config',
                    str(code/'config.json'), '--output', str(dest/'validation.csv')], check=True)
    subprocess.run([sys.executable, str(code/'analyze_predictions.py'), '--predictions',
                    str(dest/'validation.csv'), '--manifest', str(manifest), '--output', str(dest/'report')],
                   check=True)
    with (dest/'validation.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len({r['sample_id'] for r in rows}) == 466
    result = json.loads((dest/'report/report.json').read_text(encoding='utf-8'))['metrics']
    probe = json.loads((dest/'offline-check.json').read_text(encoding='utf-8'))
    assert probe['passed'] and probe['score_count'] == 40 and probe['parameters'] <= 100_000_000
    return result, probe


def write_report(runs, report):
    rows = json.loads((runs/'comparison.json').read_text(encoding='utf-8'))['comparison']
    raw = report['raw_validation']
    winner = report['selection']
    base = json.loads((V7/'metrics.json').read_text(encoding='utf-8'))['metrics']
    lines = [
        '# V8 领域适配实验结果', '',
        '两轮监督筛选，共用 V7 seed 42 的初始权重。融合比例固定为 VideoMAE 0.75、R34 0.25。', '',
        '| 组别 | 单模型准确率 | 单模型 Macro F1 | 融合准确率 | 融合 Macro F1 |',
        '|---|---:|---:|---:|---:|',
    ]
    for mode in MODES:
        single, fusion = [r for r in rows if r['mode'] == mode]
        lines.append(f"| {mode} | {single['accuracy']:.2f}% | {single['macro_f1']:.2f}% | "
                     f"{fusion['accuracy']:.2f}% | {fusion['macro_f1']:.2f}% |")
    selected = '融合' if winner['fusion'] else '单模型'
    lines += [
        '', f"选中 **{winner['mode']} {selected}**。原图复验准确率 **{raw['accuracy']:.2f}%**，"
        f"Macro F1 **{raw['macro_f1']:.2f}%**，覆盖 466 段。总推理参数 {report['parameters']:,}。",
        '', f"V7 融合参考为 {base['accuracy']:.2f}% / {base['macro_f1']:.2f}%。"
        f"准确率差 {raw['accuracy']-base['accuracy']:+.2f} 个百分点，"
        f"Macro F1 差 {raw['macro_f1']-base['macro_f1']:+.2f} 个百分点。",
        '', '公开验证集用于选型。原图复验及 Windows 下 Python 网络拦截单样本检查通过，Linux 仍待复验。',
        '', '权重 SHA-256：`' + report['weights_sha256'] + '`',
        '', '代码 ZIP SHA-256：`' + report['code_sha256'] + '`',
    ]
    (runs/'实验结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--runs', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = a.parse_args()
    runs, dest = a.runs.resolve(), a.output.resolve()
    winner = compare(runs)
    config = package(runs, dest, winner)
    result, probe = validate(dest)
    report = {'selection': winner, 'raw_validation': result, 'parameters': probe['parameters'],
              'weights_sha256': sha(dest/'model.pt'), 'code_sha256': sha(dest/'code.zip'),
              'architecture': config['architecture'], 'linux_verified': False, 'comparison_to_v7': 'public validation'}
    write(dest/'metrics.json', report)
    write_report(runs, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
