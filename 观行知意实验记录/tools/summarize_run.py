"""将真实训练、独立预测和离线检查记录汇总成可读的Markdown。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--package', required=True)
    a = p.parse_args()
    run = ROOT / a.run
    package = ROOT / a.package
    summary = load(run / 'summary.json')
    report = load(run / 'validation-report.json')
    meta = load(run / 'metadata.json')
    config = load(run / 'train-config.json')
    best = load(run / 'best_metrics.json')
    offline = load(package / 'offline-check.json')
    package_check = load(package / 'package-check.json')
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    if len(history) != config['epochs'] or any(row['train_samples_seen'] != 1995 for row in history):
        raise ValueError('训练轮次或每轮样本数量不完整')
    with (run / 'validation.csv').open(encoding='utf-8-sig', newline='') as f:
        predictions = list(csv.DictReader(f))
    if len(predictions) != 466 or not offline['passed'] or not package_check['code_and_weights_checked']:
        raise ValueError('完整预测或离线包检查未通过')
    checksum = hashlib.sha256((run / 'best_ema.pt').read_bytes()).hexdigest()
    if checksum != package_check['weights_sha256']:
        raise ValueError('模型和导出的model.pt不一致')
    text = f'''# 初版模型训练结果

模型：四通道R(2+1)D-34；公开IG65M/Kinetics400预训练后，在本届训练集微调。

## 实际结果

- 独立重新加载best_ema后，对全部466段公开验证集预测：**准确率{report['metrics']['accuracy']:.2f}%，Macro F1 {report['metrics']['macro_f1']:.2f}%**。
- 最佳轮次：第{best['epoch']}轮。训练中的混合精度验证准确率{best['accuracy']:.2f}%；以上最终指标以实际推理入口产生的CSV为准。
- 已完成{config['epochs']}轮，每轮1995段训练数据；验证集仅用于评估与检查点选择。
- 成功参数更新{summary['successful_updates']}次，FP16溢出跳过{summary['skipped_updates']}次。
- 参数量{meta['parameters']:,}，低于100M；训练峰值GPU allocated显存约{summary['peak_allocated_mib']:.0f}MiB。
- 训练与逐轮验证耗时约{summary['train_eval_sec']/60:.1f}分钟；额外预测、打包和前期缓存时间不包含在内。
- 本地逐片段推理平均耗时{report['local_pipeline_latency_ms_mean']:.1f}ms（含取帧/读图/模型计算，排除首片段）。这不是组织方正式计时成绩。

## 每轮记录

| 轮次 | 训练损失 | 验证准确率% | Macro F1% | 当轮训练秒数 |
|---|---:|---:|---:|---:|
'''
    for row in history:
        text += f"| {row['epoch']} | {row['train_loss']:.4f} | {row['accuracy']:.2f} | {row['macro_f1']:.2f} | {row['train_sec']:.1f} |\n"
    text += f'''
## 关键文件

- best_ema.pt：本次选定模型，推理配置读取它。
- inference-config.json：16帧、128×128、深度RGB+IR四通道、无检测器裁剪。
- validation.csv：完整466段预测，sample_id,predicted_class。
- validation-report.html：类别表现与错误样本，可用浏览器打开。
- validation-report.json：混淆矩阵、每类F1、错例和最终指标。
- source-code.zip / source-manifest.json / request.json：实际训练时的代码快照和配置依据。
- ../../{a.package}/model.pt：单独导出的同一模型；代码ZIP不含大权重。

模型SHA256：`{checksum}`。

训练设置：batch {config['batch']}，workers {config['workers']}，学习率{config['lr']}，seed {config['seed']}，{config['epochs']}轮。默认30轮方案未执行，本次是5轮初版。Linux参考环境尚未验证。

## 需要优先检查的类别

以下按验证集F1从低到高列出前8类，先查看错例，再决定下一次单变量实验。小样本类别的指标波动较大。

| 类别 | 验证样本数 | 召回率% | F1% |
|---|---:|---:|---:|
'''
    for row in sorted(report['per_class'], key=lambda x: x['f1'])[:8]:
        text += f"| {row['class_name']} | {row['support']} | {100*row['recall']:.1f} | {100*row['f1']:.1f} |\n"
    text += '\n此模型已经完成真实训练，可作为后续改进的基线。公开验证集用于开发，结果不等于组织方470段封闭测试成绩；正式提交前还需Linux复现、技术报告PDF、网页说明和团队材料。\n'
    (run / '初版训练结果.md').write_text(text, encoding='utf-8')
    print(json.dumps({'run': a.run, 'accuracy': report['metrics']['accuracy'],
                      'macro_f1': report['metrics']['macro_f1'], 'best_epoch': best['epoch'],
                      'weights_sha256': checksum}, ensure_ascii=False))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
