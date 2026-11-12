"""本轮续训后：固定视角/融合对照、真实图片复验、离线检查、保留最佳版本。"""
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/continue16-v5-20261106'
FUSION = ROOT/'runs/fusion-continue16-v5-20261106'
PACKAGE = ROOT/'submission/continue-v5-20261106'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def execute(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    if not (RUN/'summary.json').is_file():
        raise RuntimeError('Training must finish first')
    if FUSION.exists() or PACKAGE.exists():
        raise FileExistsError('Preserve previous evaluation; inspect before rerunning')
    execute('optimized/compare_views.py', '--run', RUN)
    execute('optimized/combine_models.py', '--r34', RUN, '--swin',
            'runs/swin192-v3-20261025', '--output', FUSION, '--r34-best-view')
    execute('optimized/evaluate.py', '--run', FUSION, '--crop',
            '--package-name', PACKAGE.name)
    import numpy as np
    cfg = read(FUSION/'config.json')
    arrays = []
    for source, component in zip(cfg['source_runs'], cfg['components']):
        mode = 'flip' if component['flip_tta'] else 'single'
        a = np.load(Path(source)/f'cached-{mode}-predictions.npz')
        arrays.append({str(k): v for k, v in zip(a['sample_ids'], a['probabilities'])})
    with (PACKAGE/'validation.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['sample_id', 'predicted_class']
        rows = list(reader)
    assert len(rows) == len({r['sample_id'] for r in rows}) == 466
    assert set(arrays[0]) == set(arrays[1]) == {r['sample_id'] for r in rows}
    for row in rows:
        p = sum(c['weight']*a[row['sample_id']] for c, a in zip(cfg['components'], arrays))
        assert int(row['predicted_class']) == int(p.argmax())
    result = read(PACKAGE/'metrics.json')
    old = read(ROOT/'runs/latest-model.json')
    scores = result['metrics']
    # analyze_predictions reports percentages; same scale as the existing pointer.
    promoted = (scores['accuracy'], scores['macro_f1']) > (old['accuracy'], old['macro_f1'])
    if promoted:
        import zipfile
        code = PACKAGE/'code'
        with zipfile.ZipFile(PACKAGE/'code.zip') as z:
            z.extractall(code)
        local_config = read(code/'config.json')
        local_config['weights_file'] = '../model.pt'
        (code/'config.json').write_text(json.dumps(local_config, indent=2), encoding='utf-8')
        new = dict(run=FUSION.relative_to(ROOT).as_posix(),
                   package=PACKAGE.relative_to(ROOT).as_posix(),
                   weights=(PACKAGE/'model.pt').relative_to(ROOT).as_posix(),
                   inference_config=(code/'config.json').relative_to(ROOT).as_posix(),
                   predictions=(PACKAGE/'validation.csv').relative_to(ROOT).as_posix(),
                   accuracy=scores['accuracy'], macro_f1=scores['macro_f1'],
                   weights_sha256=result['weights_sha256'], smoke=False)
        temp = ROOT/'runs/latest-model.json.tmp'
        temp.write_text(json.dumps(new, indent=2), encoding='utf-8')
        temp.replace(ROOT/'runs/latest-model.json')
    outcome = dict(training=read(RUN/'summary.json'), fusion_metrics=scores,
                   previous_best=old, promoted=promoted, raw_cached_predictions_match=True,
                   validation_rows=466, offline_check=read(PACKAGE/'offline-check.json'),
                   package=str(PACKAGE), linux_verified=False)
    (RUN/'outcome.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding='utf-8')
    report = (f'# 2026-11-06 续训结果\n\n'
              f'从上一轮最佳 R34 权重继续训练 6 轮，骨干学习率 5e-6，分类头 5 倍。'
              f'训练集 1995 段；验证集 466 段仅用于评估与选择。\n\n'
              f'融合准确率：{scores["accuracy"]:.4f}%；Macro F1：{scores["macro_f1"]:.4f}%。\n\n'
              f'是否更新最优版本：{promoted}。旧模型完整保留。\n\n'
              f'真实图片与缓存的 466 条预测全部一致；实际代码 ZIP 离线加载检查通过。'
              f'Linux 尚未验证。重复选择公开验证集可能造成乐观偏差，不能等同于隐藏测试成绩。\n\n'
              f'模型与代码包：{PACKAGE}\n')
    (RUN/'实验结果.md').write_text(report, encoding='utf-8')
    print(json.dumps(outcome, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
