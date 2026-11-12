"""对比完整验证结果，选择单模型方案，保留初版并生成可运行交付目录。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    names = ['stable16-single-v2-20261025', 'stable16-flip-v2-20261025',
             'crop16-single-v2-20261025', 'crop16-flip-v2-20261025']
    for name in ['crop32-single-v2-20261025', 'crop32-flip-v2-20261025',
                 'crop32-transfer-single-v2-20261025', 'crop32-transfer-flip-v2-20261025']:
        if (ROOT/'submission'/name/'metrics.json').exists():
            names.append(name)
    if (ROOT/'submission/initial-flip-v2-20261025/metrics.json').exists():
        names.append('initial-flip-v2-20261025')
    candidates = []
    for name in names:
        path = ROOT/'submission'/name
        metrics = read(path/'metrics.json')
        if not read(path/'offline-check.json')['passed']:
            raise ValueError('Offline check failed: '+name)
        candidates.append(dict(metrics, package=name))
    chosen = max(candidates, key=lambda c: (c['metrics']['accuracy'], c['metrics']['macro_f1'], -c['latency_ms']))
    if chosen['metrics']['accuracy'] < 54.72103004291846:
        raise RuntimeError('New candidate is worse than preserved initial model')
    dest = ROOT/'submission/optimized-20261025'
    shutil.copytree(ROOT/'submission'/chosen['package'], dest)
    code = dest/'code'
    with zipfile.ZipFile(dest/'code.zip') as z:
        z.extractall(code)
    shutil.copy2(dest/'model.pt', code/'model.pt')
    if chosen['person_crop']:
        shutil.copy2(code/'yolo11n.pt', dest/'yolo11n.pt')
    checksum = hashlib.sha256((dest/'model.pt').read_bytes()).hexdigest()
    if checksum != chosen['weights_sha256']:
        raise ValueError('Exported model hash mismatch')
    with (dest/'validation.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 466 or len({x['sample_id'] for x in rows}) != 466:
        raise ValueError('Final CSV does not contain 466 unique samples')
    epochs = {}
    for name in ['stable16-v2-20261025', 'crop16-v2-20261025', 'crop32-v2-20261025']:
        if (ROOT/'runs'/name/'summary.json').exists():
            epochs[name] = read(ROOT/'runs'/name/'summary.json')
    baseline = read(ROOT/'runs/initial16-20261020-5ep/validation-report.json')['metrics']
    report = '# 本机优化训练结果 2026年10月25日\n\n'
    report += f"最终选择 **{chosen['package']}**。完整466段公开验证集准确率 **{chosen['metrics']['accuracy']:.2f}%**，Macro F1 **{chosen['metrics']['macro_f1']:.2f}%**。\n\n"
    report += f"相对初版54.72%/43.44%，准确率提高{chosen['metrics']['accuracy']-baseline['accuracy']:.2f}个百分点，Macro F1提高{chosen['metrics']['macro_f1']-baseline['macro_f1']:.2f}个百分点。\n\n"
    report += '## 完整对照\n\n| 方案 | 准确率% | Macro F1% | 本机平均毫秒/段 | 参数量 |\n|---|---:|---:|---:|---:|\n'
    report += '| 初版5轮，单次推理 | 54.72 | 43.44 | 113.3 | 63,514,700 |\n'
    for c in candidates:
        report += f"| {c['package']} | {c['metrics']['accuracy']:.2f} | {c['metrics']['macro_f1']:.2f} | {c['latency_ms']:.1f} | {c['parameters']:,} |\n"
    report += '\n## 训练与实现\n\n'
    for name, stats in epochs.items():
        report += f"- {name}：完成{stats['epochs_completed']}轮，最佳第{stats['best_epoch']}轮，实际更新{stats['steps']}次，耗时约{stats['elapsed_sec']/60:.1f}分钟。\n"
    report += '\n16帧两组从本团队初版权重微调；32帧组从16帧裁剪组最佳权重继续微调。训练数据每轮1995段，验证集不更新参数。新增BF16和channels_last_3d、分层学习率、慢BN统计、温和类别加权、标签平滑、EMA。人物裁剪使用官方配套YOLO11n在IR上定位，深度与IR共享裁剪。未使用额外RGB/骨架或范围外CUHK数据。\n\n'
    report += '名字含transfer的对照复用16帧组权重，仅改变推理帧数，没有额外更新参数；不能把该项收益归因于32帧训练。\n\n'
    report += '这里比较的是组合训练配方与裁剪路线，不能把收益分别归因于某一个正则化组件。验证集已用于开发和模型选择，并非独立测试成绩。\n\n'
    report += f"最终仅使用一个分类器，翻转视角共享参数；总参数{chosen['parameters']:,}，不超过100M。平均推理延迟是本机实测，首样本预热不计，不是赛事正式计时结果。\n\n"
    report += '## 使用模型\n\n权重和代码位于submission/optimized-20261025：model.pt、code.zip、完整validation.csv。code目录已解压并放入同一权重，可直接运行：\n\n```powershell\n.\\.venv\\Scripts\\python.exe submission/optimized-20261025/code/predict.py --dataset data/raw/validation --config submission/optimized-20261025/code/config.json --output runs/optimized-recheck.csv\n```\n\n'
    report += f'最终模型SHA256：`{checksum}`。代码ZIP不包含大分类器权重；若采用人物裁剪，5.6MB的YOLO11n在代码ZIP中，实际参数一并计数。\n\n'
    if chosen['person_crop']:
        report += 'QQ交付权重时一并发送根目录model.pt与yolo11n.pt，并分别说明动作分类器和人物定位器用途；运行时用同一份代码/配置。\n\n'
    mode = 'flip' if chosen['flip_tta'] else 'single'
    report_target = chosen.get('report_file', f"{chosen['run']}/validation-{mode}-report.html")
    report += f"[最终模型逐类表现与错例]({Path(report_target).relative_to('runs').as_posix()})。\n\n"
    selected_analysis = read((ROOT/report_target).with_suffix('.json'))
    original_analysis = read(ROOT/'runs/initial16-20261020-5ep/validation-report.json')
    improvements = [(new['f1']-old['f1'], old, new) for old, new in zip(original_analysis['per_class'], selected_analysis['per_class'])]
    report += '## 类别表现变化\n\n以下为F1提升最多的5类；样本量很小的类别需谨慎解读。\n\n| 类别 | 验证样本 | 初版F1% | 新版F1% |\n|---|---:|---:|---:|\n'
    for _, old, new in sorted(improvements, key=lambda row: row[0], reverse=True)[:5]:
        report += f"| {new['class_name']} | {new['support']} | {old['f1']*100:.1f} | {new['f1']*100:.1f} |\n"
    report += '\n当前F1最低的类别：'+', '.join(x['class_name'] for x in sorted(selected_analysis['per_class'], key=lambda x: x['f1'])[:5])+'。可从完整错例报告继续分析，不应只提高训练集分数。\n\n'
    report += '## 来源与边界\n\n- PyTorch/torchvision：https://github.com/pytorch/vision\n- timm EMA：https://github.com/huggingface/pytorch-image-models\n- Ultralytics YOLO11n：https://github.com/ultralytics/ultralytics/tree/v8.3.228\n- TSN优化策略参考：https://github.com/yjxiong/tsn-pytorch\n- IG65M通用预训练：https://github.com/moabitcoin/ig65m-pytorch\n\n许可证、模型来源、SHA256和具体用途见competition/optimization-v2-sources.json。运行依赖快照见environment/requirements-v2-lock.txt。人物裁剪所用依赖沿用官方参考版本，本机opencv-python与参考headless构建差别需在Linux复验时确认。Linux参考环境尚未验证；正式技术报告PDF、网页说明和QQ权重发送仍需完成。\n'
    report_path = ROOT/'runs/优化训练结果_20261025.md'
    report_path.write_text(report, encoding='utf-8')
    write(ROOT/'runs/optimization-v2-comparison.json', {'selected': chosen, 'candidates': candidates, 'training': epochs, 'linux_verified': False})
    write(ROOT/'runs/latest-model.json', {'run': chosen['run'], 'package': 'submission/optimized-20261025', 'weights': 'submission/optimized-20261025/model.pt',
          'inference_config': 'submission/optimized-20261025/code/config.json', 'predictions': 'submission/optimized-20261025/validation.csv',
          'accuracy': chosen['metrics']['accuracy'], 'macro_f1': chosen['metrics']['macro_f1'], 'weights_sha256': checksum, 'smoke': False})
    text = (ROOT/'README.md').read_text(encoding='utf-8')
    heading = '## 最新优化模型 2026年10月25日\n\n'
    heading += f"完整验证准确率 **{chosen['metrics']['accuracy']:.2f}%**，Macro F1 **{chosen['metrics']['macro_f1']:.2f}%**。\n\n"
    heading += '[对照实验与运行说明](runs/优化训练结果_20261025.md)。权重、代码ZIP、完整CSV和可运行code目录在`submission/optimized-20261025`；初版仍保留。Linux参考环境尚未验证。\n\n'
    (ROOT/'README.md').write_text(text.replace('## 已有初版模型', heading+'## 已有初版模型', 1), encoding='utf-8')
    info = read(ROOT/'runs/verification-summary.json')
    info.update(last_updated='2026-10-25', latest_run=chosen['run'], latest_package='submission/optimized-20261025', accuracy=chosen['metrics']['accuracy'],
                macro_f1=chosen['metrics']['macro_f1'], optimization_epochs_completed=sum(c['epochs_completed'] for c in epochs.values()),
                optimization_training_steps=sum(c['steps'] for c in epochs.values()),
                full_validation_done=True, linux_verified=False, formal_submission_ready=False)
    write(ROOT/'runs/verification-summary.json', info)
    print(json.dumps({'selected': chosen['package'], 'accuracy': chosen['metrics']['accuracy'],
                      'macro_f1': chosen['metrics']['macro_f1'], 'report': str(report_path)}, ensure_ascii=False))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
