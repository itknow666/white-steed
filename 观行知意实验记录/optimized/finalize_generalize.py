"""对第二阶段探索进行审计，只用完整图片复验的结果更新交付模型。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runs = ['frozenbn16-v3-20261025', 'temporal16-v3-20261025', 'detail192-v3-20261025']
    if (ROOT/'runs/swin192-v3-20261025/summary.json').exists():
        runs.append('swin192-v3-20261025')
    if (ROOT/'runs/ircontrast16-v3-20261025/summary.json').exists():
        runs.append('ircontrast16-v3-20261025')
    views, training = [], {}
    for name in runs:
        training[name] = read(ROOT/'runs'/name/'summary.json')
        views.extend(read(ROOT/'runs'/name/'cached-view-comparison.json'))
    cached_best = max(views, key=lambda x: (x['accuracy'], x['macro_f1']))
    old = read(ROOT/'submission/optimized-20261025/metrics.json')
    candidates = [dict(old, package='optimized-20261025')]
    for package in sorted((ROOT/'submission').glob('*-v3-20261025')):
        if (package/'metrics.json').exists():
            if not read(package/'offline-check.json')['passed']:
                raise ValueError('Offline check failed')
            candidates.append(dict(read(package/'metrics.json'), package=package.name))
    if len(candidates) < 2:
        raise ValueError('Need independent full-image verification of at least one v3 candidate')
    chosen = max(candidates, key=lambda x: (x['metrics']['accuracy'], x['metrics']['macro_f1'], -x['latency_ms']))
    dest = ROOT/'submission/optimized-generalize-20261025'
    shutil.copytree(ROOT/'submission'/chosen['package'], dest, ignore=shutil.ignore_patterns('code'))
    with zipfile.ZipFile(dest/'code.zip') as z:
        entries = z.infolist()
        assert len(entries) <= 256 and sum(x.file_size for x in entries) <= 64*1024**2
        assert (dest/'code.zip').stat().st_size <= 20*1024**2
        assert {'inference.py','config.json'} <= set(z.namelist())
        z.extractall(dest/'code')
    shutil.copy2(dest/'model.pt', dest/'code/model.pt')
    if chosen['person_crop']:
        shutil.copy2(dest/'code/yolo11n.pt', dest/'yolo11n.pt')
    assert sha(dest/'model.pt') == chosen['weights_sha256']
    assert sha(dest/'code.zip') == chosen['code_sha256']
    with (dest/'validation.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['sample_id','predicted_class']
        rows = list(reader)
    manifest = read(ROOT/'data/raw/validation/manifest.json')
    labels = {s.get('sample_id',s['file']+'.npy'):int(s['label']) for s in manifest['samples']}
    assert len(rows) == len(labels) == len({r['sample_id'] for r in rows}) == 466
    assert {r['sample_id'] for r in rows} == set(labels)
    cm = [[0]*40 for _ in range(40)]
    for row in rows:
        predicted = int(row['predicted_class'])
        assert 0 <= predicted < 40
        cm[labels[row['sample_id']]][predicted] += 1
    accuracy = 100*sum(cm[i][i] for i in range(40))/466
    f1 = 100*sum(2*cm[i][i]/max(1,sum(cm[i])+sum(cm[j][i] for j in range(40))) for i in range(40))/40
    assert abs(accuracy-chosen['metrics']['accuracy']) < 1e-8
    assert abs(f1-chosen['metrics']['macro_f1']) < 1e-8
    assert chosen['parameters'] <= 100_000_000
    report = '# 继续优化实验：泛化、时序与分辨率\n\n'
    report += f"最终选择 **{chosen['package']}**：完整466段公开验证准确率 **{accuracy:.2f}%**，Macro F1 **{f1:.2f}%**。相对初版54.72%/43.44%，分别提高{accuracy-54.72103004291846:.2f}和{f1-43.44292042521556:.2f}个百分点。\n\n"
    report += '原有优化模型和各实验权重均保留。本报告的“缓存对照”用于开发选择；交付版本必须另外读取真实图片、运行实际赛事接口、生成完整CSV并进行离线加载检查。\n\n'
    report += '## 本轮做了什么\n\n'
    report += '- 固定BN运行统计、保留仿射参数学习，降低小数据微调的统计变化；与TSN原版partialBN并非完全相同。\n- 从32帧缓存的16个时间段各随机取一帧，深度与IR同步；验证固定为官方16帧采样。\n- 温和Mixup（alpha=0.2、每批50%概率）与10%概率小区域遮挡；只作用于训练数据，不做不适合伪彩深度的RGB色相增强。\n- 将16帧裁剪输入从128×128增至192×192，检查手部/小物体细节是否有帮助。\n\n'
    report += '前三组从同一份本团队crop16最佳权重出发，骨干学习率1e-5、头部5倍；保留EMA/BF16/channels_last_3d。各组同时涉及部分训练配方变化，不能把差异归因于单个组件；单随机种子的探索结果也不是严格统计结论。精确参数见每个run/config.json和runs/optimization-v3-plan.json。Mixup下训练准确率是混合标签加权的诊断值，不能与未混合训练直接比较。\n\n'
    if 'swin192-v3-20261025' in training:
        report += '第四组改用TorchVision Video Swin-T，从官方通用Kinetics400权重迁移，四通道投影加40类新头；骨干学习率1e-4、分类头10倍。此组不是从CUHK外部权重初始化。它测试不同架构是否有互补特征；公开预训练来源与SHA256在weights/swin3d_t-7615ae03.source.json。\n\n'
    if 'ircontrast16-v3-20261025' in training:
        report += '查看真实错例发现部分IR很暗，因此补充第五组：在每个片段内按IR像素1%/99%分位做固定亮度拉伸，全片段共用映射，深度通道不变；从crop16模型微调，BN统计继续适应，没有Mixup/遮挡。只使用当前片段自身像素，不拟合验证/测试标签。原始数据保留，派生缓存另存。\n\n'
    report += '## 真实训练记录\n\n| 实验 | 完成轮数 | 最佳轮 | 更新次数 | 训练分钟 | 最佳单视角准确率% |\n|---|---:|---:|---:|---:|---:|\n'
    for name, s in training.items():
        report += f"| {name} | {s['epochs_completed']} | {s['best_epoch']} | {s['steps']} | {s['elapsed_sec']/60:.1f} | {s['accuracy']:.2f} |\n"
    report += '\n## 缓存视角对照\n\n| 实验 | 翻转平均 | 准确率% | Macro F1% |\n|---|---|---:|---:|\n'
    for c in views:
        report += f"| {Path(c['run']).name} | {c['flip_tta']} | {c['accuracy']:.2f} | {c['macro_f1']:.2f} |\n"
    report += '\n## 真实图片与赛事接口复验\n\n| 版本 | 准确率% | Macro F1% | 毫秒/段 |\n|---|---:|---:|---:|\n'
    for c in candidates:
        report += f"| {c['package']} | {c['metrics']['accuracy']:.2f} | {c['metrics']['macro_f1']:.2f} | {c['latency_ms']:.1f} |\n"
    if (ROOT/'runs/fusion-v3-20261025/cached-fusion-comparison.json').exists():
        fusion = read(ROOT/'runs/fusion-v3-20261025/cached-fusion-comparison.json')
        report += '\n额外检查R34与Swin-T概率平均，固定Swin权重0.25/0.5/0.75，人物定位器共用；这只是验证集上的模型选择，不增加训练更新。缓存对照如下，只有上表出现fusion的版本才完成真实图片复验：\n\n'
        for c in fusion['comparisons']:
            report += f"- Swin权重{c['swin_weight']}：准确率{c['accuracy']:.2f}%，Macro F1 {c['macro_f1']:.2f}%。\n"
        report += f"\n组合方案共{fusion['parameters']:,}参数，未使用两个完整R34叠加。\n\n"
    report += f"\n交付文件：submission/optimized-generalize-20261025。model.pt SHA256：`{chosen['weights_sha256']}`；总参数{chosen['parameters']:,}，代码ZIP {len(entries)}文件、{(dest/'code.zip').stat().st_size:,}字节，符合本地检查限制。人物定位器yolo11n.pt同目录另附，也在代码ZIP内；提交权重时要同时保留。\n\n"
    report += '## 继续使用\n\n```powershell\n.\\.venv\\Scripts\\python.exe submission/optimized-generalize-20261025/code/predict.py --dataset data/raw/validation --config submission/optimized-generalize-20261025/code/config.json --output runs/generalize-recheck.csv\n```\n\n'
    report += 'VS Code任务“优化模型：完整验证集预测”指向本次最终目录。不要把不同实验的代码、配置和权重混用。\n\n'
    report += '官网代码提交使用现成code.zip；不要直接压缩code目录，那里为本地运行另外放入了大权重。model.pt内含最终分类器参数，人物定位器为yolo11n.pt；按赛事要求另行交付对应权重。\n\n'
    report += '## 公开方案来源与限制\n\n- [TSN的BN策略](https://github.com/yjxiong/tsn-pytorch/blob/master/models.py)：启发统计冻结，未加载TSN权重。\n- [PyTorchVideo分类教程](https://pytorchvideo.org/docs/tutorial_classification)：训练随机采样与验证固定采样。\n- [TorchVision训练配方](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)：Mixup、低概率遮挡和分辨率调优。其ImageNet收益不代表本题收益。\n\n'
    report += '- [TorchVision Video Swin-T](https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py)：不同模型架构与官方通用预训练；BSD-3-Clause。TSM也已调研，但本轮没有训练TSM模型。\n\n'
    report += '当前分数来自已用于模型选择的公开验证集，不是隐藏测试成绩；未使用测试数据、额外模态或额外CUHK片段。Linux参考环境尚未验证，Python网络连接拦截不替代正式断网环境。正式PDF和赛事提交尚未执行。\n'
    report += '\n## 错例诊断证据\n\n此前66.31%模型的常见混淆包括喝水/吃东西、读文件/翻页、弓步/深蹲、玩游戏/用手机。以下为真实验证错例的同步深度与IR预览；部分IR很暗，读文件例中的人也很小。这些是改进假设的线索，不能据此认定全部错误原因。\n\n'
    report += '![真实错例同步画面](confusion-examples-v2.png)\n\n![IR增强前后](ir-contrast-preview-v3.png)\n\n'
    if chosen.get('report_file'):
        report += f"[交付模型的完整逐类指标和错例]({Path(chosen['report_file']).relative_to('runs').as_posix()})。\n"
    path = ROOT/'runs/继续优化实验_20261025.md'
    path.write_text(report, encoding='utf-8')
    write(ROOT/'runs/optimization-v3-comparison.json', {'selected':chosen,'cached_views':views,'cached_best':cached_best,'full_image_candidates':candidates,'training':training,'linux_verified':False})
    package = dest.relative_to(ROOT).as_posix()
    write(ROOT/'runs/latest-model.json', {'run':chosen['run'],'package':package,'weights':package+'/model.pt','inference_config':package+'/code/config.json','predictions':package+'/validation.csv','accuracy':accuracy,'macro_f1':f1,'weights_sha256':chosen['weights_sha256'],'smoke':False})
    info = read(ROOT/'runs/verification-summary.json')
    info.update(latest_run=chosen['run'],latest_package=package,accuracy=accuracy,macro_f1=f1,
                generalization_epochs_completed=sum(s['epochs_completed'] for s in training.values()),
                generalization_training_steps=sum(s['steps'] for s in training.values()),linux_verified=False,formal_submission_ready=False)
    write(ROOT/'runs/verification-summary.json',info)
    readme = ROOT/'README.md'
    content = readme.read_text(encoding='utf-8')
    content = content.replace('## 最新优化模型 2026年10月25日',f'## 继续优化后的最终模型\n\n完整验证准确率 **{accuracy:.2f}%**，Macro F1 **{f1:.2f}%**。最终包在`{package}`；[新增探索与复验](runs/继续优化实验_20261025.md)。\n\n## 上一阶段优化模型 2026年10月25日',1)
    readme.write_text(content,encoding='utf-8')
    task_path = ROOT/'.vscode/tasks.json'
    tasks = read(task_path)
    for task in tasks['tasks']:
        if task['label']=='优化模型：完整验证集预测':
            task['args']=[x.replace('submission/optimized-20261025','submission/optimized-generalize-20261025') for x in task['args']]
    write(task_path,tasks)
    print(json.dumps({'selected':chosen['package'],'accuracy':accuracy,'macro_f1':f1,'report':str(path)},ensure_ascii=False))


if __name__ == '__main__':
    main()
