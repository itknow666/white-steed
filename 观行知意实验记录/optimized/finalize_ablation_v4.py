"""真实图片复验、固定类别指标、离线加载检查通过后才更新最佳版本并打包。"""
import csv
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ablation-v4-20261027'
sys.path.insert(0,str(Path(__file__).parent))
from train import write,sha,metrics


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def command(args,name):
    with (OUT/(name+'.log')).open('a',encoding='utf-8') as log:
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def verify(package,expected,ids):
    meta=read(package/'metrics.json')
    check=read(package/'offline-check.json')
    if not check['passed'] or check['score_count']!=40 or not check['finite']:
        raise ValueError('Offline interface verification failed')
    if meta['parameters']>100_000_000 or meta['weights_sha256']!=sha(package/'model.pt'):
        raise ValueError('Parameter count or model digest failed')
    with (package/'validation.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['sample_id','predicted_class']: raise ValueError('CSV header')
        rows=list(reader)
    actual={r['sample_id']:int(r['predicted_class']) for r in rows}
    if len(rows)!=466 or len(actual)!=466 or set(actual)!=set(ids): raise ValueError('CSV IDs')
    if not all(0<=v<40 for v in actual.values()): raise ValueError('Class range')
    if [actual[s] for s in ids]!=expected.argmax(1).tolist():
        raise ValueError('Live images disagree with cached predictions')
    return meta


def diagnostics(result, package, labels, ids):
    def predictions(path):
        with path.open(encoding='utf-8-sig',newline='') as f:
            return {r['sample_id']:int(r['predicted_class']) for r in csv.DictReader(f)}
    old=predictions(ROOT/'submission/optimized-generalize-20261025/validation.csv')
    new=predictions(package/'validation.csv')
    a=np.array([old[s] for s in ids]);b=np.array([new[s] for s in ids])
    ca=np.bincount(labels*40+a,minlength=1600).reshape(40,40)
    cb=np.bincount(labels*40+b,minlength=1600).reshape(40,40)
    changes={'old_correct':int((a==labels).sum()),'new_correct':int((b==labels).sum()),
             'fixed_old_errors':int(((a!=labels)&(b==labels)).sum()),
             'lost_old_correct':int(((a==labels)&(b!=labels)).sum()),
             'net_correct_gain':int((b==labels).sum()-(a==labels).sum()),
             'sample_count':466,'not_independent_test':True}
    write(OUT/'paired-comparison.json',changes)
    classes=read(ROOT/'data/raw/validation/manifest.json')['classes']
    with (OUT/'per-class-comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['class_id','class_name','support','old_recall','new_recall','old_f1','new_f1'])
        for i in range(40):
            support=int(ca[i].sum())
            w.writerow([i,classes[i],support,float(ca[i,i]/max(1,support)),float(cb[i,i]/max(1,support)),
                        float(2*ca[i,i]/max(1,ca[i].sum()+ca[:,i].sum())),float(2*cb[i,i]/max(1,cb[i].sum()+cb[:,i].sum()))])
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for row in result['groups']:
        history=[json.loads(line) for line in (Path(row['run'])/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        for ax,key in zip(axes,['accuracy','macro_f1']):
            ax.plot([h['epoch'] for h in history],[h[key] for h in history],marker='o',label=row['name'])
            ax.set(xlabel='Epoch',ylabel=key+' (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(OUT/'learning-curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True)
    image=ax.imshow(cb,cmap='Blues');fig.colorbar(image,ax=ax,label='Clips')
    ax.set(xlabel='Predicted class ID',ylabel='True class ID',title='New candidate: validation confusion matrix',xticks=range(0,40,2),yticks=range(0,40,2))
    fig.savefig(OUT/'confusion-matrix.png',dpi=160);plt.close(fig)
    return changes


def main():
    result=read(OUT/'results.json')
    views=[]
    for row in result['groups']:
        for view in read(Path(row['run'])/'cached-view-comparison.json'):
            views.append(dict(view,group=row['name']))
    best=max(views,key=lambda v:(v['accuracy'],v['macro_f1']))
    run=Path(best['run'])
    config=read(run/'config.json')
    mode='flip' if best['flip_tta'] else 'single'
    prediction=np.load(run/f'cached-{mode}-predictions.npz')
    ids=prediction['sample_ids'].tolist()
    single_package=ROOT/'submission/ablation-v4-single-20261027'
    if not (single_package/'offline-check.json').is_file():
        args=['optimized/evaluate.py','--run',run,'--crop','--package-name',single_package.name,'--check-cache',config['cache']]
        if best['flip_tta']: args.append('--flip')
        command(args,'verify-single')
    single=verify(single_package,prediction['probabilities'],ids)
    fusion_candidates=[]
    for row in result['groups']:
        candidate_run=ROOT/'runs'/f"fusion-{row['name']}-v4-20261027"
        if not (candidate_run/'config.json').is_file():
            command(['optimized/combine_models.py','--r34',row['run'],'--swin',ROOT/'runs/swin192-v3-20261025',
                     '--output',candidate_run,'--r34-best-view'],'combine-'+row['name'])
        comparison=read(candidate_run/'cached-fusion-comparison.json')
        fusion_candidates.append({'group':row['name'],'run':str(candidate_run),'source_run':row['run'],**comparison})
    write(OUT/'fusion-comparisons.json',fusion_candidates)
    selected_fusion=max(fusion_candidates,key=lambda r:(r['best']['accuracy'],r['best']['macro_f1']))
    fusion_run=Path(selected_fusion['run'])
    fusion_config=read(fusion_run/'config.json')
    fusion_source=Path(selected_fusion['source_run'])
    fusion_source_config=read(fusion_source/'config.json')
    fusion_view='flip' if fusion_config['components'][0]['flip_tta'] else 'single'
    fusion_prediction=np.load(fusion_source/f'cached-{fusion_view}-predictions.npz')
    fusion_order={s:i for i,s in enumerate(fusion_prediction['sample_ids'].tolist())}
    jx=[fusion_order[s] for s in ids]
    if not np.array_equal(prediction['labels'],fusion_prediction['labels'][jx]): raise ValueError('R34 fusion label order')
    comparisons=read(fusion_run/'cached-fusion-comparison.json')
    alpha=comparisons['best']['swin_weight']
    other=np.load(ROOT/'runs/swin192-v3-20261025/cached-single-predictions.npz')
    other_order={s:i for i,s in enumerate(other['sample_ids'].tolist())}
    ix=[other_order[s] for s in ids]
    if not np.array_equal(prediction['labels'],other['labels'][ix]): raise ValueError('Fusion label order')
    expected=(1-alpha)*fusion_prediction['probabilities'][jx]+alpha*other['probabilities'][ix]
    fusion_package=ROOT/'submission/ablation-v4-fusion-20261027'
    if not (fusion_package/'offline-check.json').is_file():
        command(['optimized/evaluate.py','--run',fusion_run,'--crop','--package-name',fusion_package.name,
                 '--check-cache',fusion_source_config['cache']],'verify-fusion')
    fusion=verify(fusion_package,expected,ids)
    old=read(ROOT/'submission/optimized-generalize-20261025/metrics.json')
    candidates=[(single,single_package,run),(fusion,fusion_package,fusion_run)]
    chosen,package,chosen_run=max(candidates,key=lambda x:(x[0]['metrics']['accuracy'],x[0]['metrics']['macro_f1']))
    improved=(chosen['metrics']['accuracy'],chosen['metrics']['macro_f1'])>(old['metrics']['accuracy'],old['metrics']['macro_f1'])
    changes=diagnostics(result,package,prediction['labels'],ids)
    verified={'old':old['metrics'],'single':single['metrics'],'fusion':fusion['metrics'],
              'new_best_improved':improved,'new_candidate':str(package),'all466_live_cached_predictions_identical':True,
              'csv_ids_classes_valid':True,'offline_zip_reload_passed':True,'parameters':chosen['parameters'],
              'crop_policies':read(fusion_run/'config.json')['components'],'linux_verified':False,'test_accessed':False}
    write(OUT/'verification.json',verified)
    # 只有通过完整图片和离线检查且分数更好的候选才替换最新指针。
    if improved:
        with zipfile.ZipFile(package/'code.zip') as z:
            runtime=package/'code';runtime.mkdir(exist_ok=True);z.extractall(runtime)
        local_config=read(runtime/'config.json')
        local_config['weights_file']='../model.pt'
        write(runtime/'config.json',local_config)
        write(ROOT/'runs/latest-model.json',{'run':chosen_run.relative_to(ROOT).as_posix(),
              'package':package.relative_to(ROOT).as_posix(),'weights':(package/'model.pt').relative_to(ROOT).as_posix(),
              'inference_config':(package/'code/config.json').relative_to(ROOT).as_posix(),
              'predictions':(package/'validation.csv').relative_to(ROOT).as_posix(),**chosen['metrics'],
              'weights_sha256':chosen['weights_sha256'],'smoke':False})
    lines=['# 第四轮优化实验：裁剪、分阶段微调与学习率对照','',
           '只使用比赛训练集1995段更新参数，公开验证集466段用于评估和方法选择。未读取隐藏测试数据。',
           '每组从相同crop16-v2最佳权重出发，seed42、6轮、batch8、16帧128尺寸；固定其余设置。',
           '扩大框1.4→1.8复用相同IR检测框；两种模态使用同一窗口。冻结组前2轮冻结stem/layer1/layer2及其BN统计，后4轮解冻且学习率再乘0.5。',
           '学习率两组跟随前面选定的裁剪和训练策略。此顺序比较并非全部因素的完整网格实验。','',
           '| 实验 | 扩框系数 | 冻结前层轮数 | 基础学习率 | 最佳轮 | 单视角准确率 | 单视角Macro F1 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['groups']:
        c=read(Path(row['run'])/'config.json')
        lines.append(f"| {row['name']} | {c['crop_scale']} | {row['frozen']} | {row['lr']:g} | {row['best_epoch']} | {row['accuracy']:.2f}% | {row['macro_f1']:.2f}% |")
    lines+=['','## 视角与融合',
            f"本轮最佳单模型：{best['group']}，{'翻转平均' if best['flip_tta'] else '单视角'}。真实图片复验准确率{single['metrics']['accuracy']:.2f}%，Macro F1 {single['metrics']['macro_f1']:.2f}%。",
            '五组各自选择已记录的最佳单/翻转视角，与原Swin-T按预先固定的0.25/0.5/0.75比例共比较15种融合；全部结果保存在fusion-comparisons.json。额外融合方案在计算新分数前登记于fusion-plan.json，未进行参数更新。',
            f"最佳融合来自{selected_fusion['group']}，Swin权重{alpha}，真实图片复验准确率{fusion['metrics']['accuracy']:.2f}%，Macro F1 {fusion['metrics']['macro_f1']:.2f}%。",
            '融合共享一次检测，每个分类器使用自己训练时的扩框系数，避免缓存和实际推理不一致。',
            f"旧最佳：{old['metrics']['accuracy']:.2f}% / {old['metrics']['macro_f1']:.2f}%。"+
            ('新候选通过校验并更新latest-model.json。' if improved else '本轮未超过旧最佳，latest-model.json继续指向旧模型。'),
            f"候选修正旧错误{changes['fixed_old_errors']}段，同时将原本正确的{changes['lost_old_correct']}段预测错，净变化{changes['net_correct_gain']:+d}段。",
            '', '## 校验与局限',
            f"五组共完成{result['epochs_total']}轮；保存每组权重、断点、逐轮日志、配置和全部视角结果。",
            '两个新候选均从实际图片预测全部466段，与缓存预测类别逐条一致；CSV列名、样本ID、遗漏重复、0—39范围均检查通过。',
            '实际代码ZIP在新进程中阻断Python网络连接后成功加载，返回40个有限分数；参数与代码包大小限制检查通过。该检查不是系统级网络隔离。',
            '冻结/解冻的梯度和BN状态已独立验证。尚未实际运行官方Linux环境，正式报告和隐藏测试结果尚无。',
            '固定了随机种子，但启用cuDNN benchmark和TF32，未要求严格确定性；跨硬件重新训练不保证逐位一致。最终保存权重的推理流程已独立复验。',
            '多轮使用公开验证集做选择会产生选择偏差；少量样本的涨分不等于隐藏测试改善，不宣称统计显著。',
            '各组起点已经在1.4扩框上微调过；本轮比较的是6轮预算内的续训效果，并未从通用预训练分别重训两种裁剪策略。扩大框可能需要更长适应期，不能据此判定该策略普遍无效。',
            '训练样本存在明显类别不均衡：看电视仅6段，行走232段；验证集中看电视、叠衣服各仅3段。弱类别的单段变化会明显影响该类F1，不能只看总体准确率。类别权重只用训练集计数计算。',
            '本轮没有尝试UniFormer、蒸馏、ECA或替换激活函数。当前先完成约定的受控实验，再依据结果决定是否值得开展更复杂路线。',
            '', '## 文件',f"新单模型：`{single_package.relative_to(ROOT)}`。",f"新融合模型：`{fusion_package.relative_to(ROOT)}`。",
            '预先登记计划plan.json，阶段验证stage-check.json，裁剪审查cache/wide16-v4-20261027/audit.json及crop-comparison.png。',
            '每组config.json和history.jsonl记录实际参数；训练与恢复入口optimized/run_ablation_v4.py。']
    report=OUT/'实验结果.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # 与旧桌面包分开保存，内容明确标为本轮候选，避免覆盖之前可用版本。
    stage=ROOT/'submission/观行知意_第四轮模型与源码_20261027'
    stage.mkdir(exist_ok=True)
    for name in ['code.zip','model.pt','validation.csv','metrics.json','offline-check.json']:
        shutil.copy2(package/name,stage/name)
    with zipfile.ZipFile(package/'code.zip') as z:
        runtime=stage/'运行代码';runtime.mkdir(exist_ok=True)
        z.extractall(runtime)
    shutil.copy2(ROOT/'baseline/yolo11n.pt',stage/'yolo11n.pt')
    shutil.copy2(report,stage/'实验结果.md')
    shutil.copy2(OUT/'verification.json',stage/'verification.json')
    source=stage/'开发源码'
    for folder in ['optimized','baseline','environment','.vscode']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py','.json','.md','.txt','.ps1']:
                target=source/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    provenance=list((ROOT/'weights').glob('*.json'))+list((ROOT/'competition').glob('*sources*.json'))+list((ROOT/'competition').glob('*LICENSE*.txt'))
    for path in provenance:
        target=source/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for row in result['groups']:
        for name in ['config.json','history.jsonl','summary.json','cached-view-comparison.json']:
            target=source/'runs'/Path(row['run']).name/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(row['run'])/name,target)
    for name in ['plan.json','fusion-plan.json','fusion-comparisons.json','stage-check.json','crop-routing-check.json','class-counts.json','results.json']:
        shutil.copy2(OUT/name,stage/name)
    for name in ['paired-comparison.json','per-class-comparison.csv','learning-curves.png','confusion-matrix.png']:
        shutil.copy2(OUT/name,stage/name)
    shutil.copy2(ROOT/'cache/wide16-v4-20261027/crop-comparison.png',stage/'裁剪对照.png')
    status='本轮候选优于旧最佳，已更新项目最新模型指针。' if improved else '本轮候选未超过旧最佳；正式选择仍保留旧版67.38%融合模型。'
    (stage/'README_先看这里.md').write_text(f'''# 第四轮模型与源码

{status}

本包候选准确率：{chosen['metrics']['accuracy']:.2f}%，Macro F1：{chosen['metrics']['macro_f1']:.2f}%，仅为公开验证集结果。
model.pt是本包实际候选权重，code.zip是配套赛事代码包；validation.csv为466段预测。
全部源码和五组实验配置在开发源码，细节见实验结果.md。原始数据、缓存、Python环境、训练断点未放入本包，仍保留在E:/SNN/wujie-action。

在本机项目环境中运行（PowerShell，先进入本包解压后的目录）：

```powershell
Copy-Item ./model.pt ./运行代码/model.pt
& E:/SNN/wujie-action/.venv/Scripts/python.exe ./运行代码/predict.py --dataset E:/SNN/wujie-action/data/raw/validation --config ./运行代码/config.json --output ./复现预测.csv
```

如需复训，请使用原项目结构和optimized/run_ablation_v4.py，依赖快照见environment。已有完成结果不会重复训练；恢复须保留resume.pt。
完整重复本轮训练还需要原项目runs/crop16-v2-20261025/best_ema.pt作为共同起点，本包没有重复打包该历史初始化权重；它仍保留在原项目中。本包model.pt可以独立用于推理。
官网只提交code.zip和CSV/报告PDF，模型权重按赛事要求单独交付，不上传整个大压缩包。正式技术报告PDF与Linux复验尚未完成。
''',encoding='utf-8')
    files=sorted(p for p in stage.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json')
    write(stage/'SHA256SUMS.json',{p.relative_to(stage).as_posix():sha(p) for p in files})
    encoded=subprocess.check_output(['powershell','-NoProfile','-Command',
            "[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes([Environment]::GetFolderPath('Desktop')))"],text=True,encoding='ascii').strip()
    desktop=Path(base64.b64decode(encoded).decode('utf-16-le'))
    archive=desktop/(stage.name+'.zip')
    if archive.exists(): raise FileExistsError(f'Preserve existing desktop package: {archive}')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(p for p in stage.rglob('*') if p.is_file()):
            z.write(path,stage.name+'/'+path.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('ZIP CRC failed')
        digests=read(stage/'SHA256SUMS.json')
        import hashlib
        for name,digest in digests.items():
            with z.open(stage.name+'/'+name) as f:
                h=hashlib.sha256()
                for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
            if h.hexdigest()!=digest: raise ValueError('Archive digest mismatch')
    write(OUT/'delivery.json',dict(verified,desktop_archive=str(archive),bytes=archive.stat().st_size,
                                 all_file_hashes_verified=True,model_sha256=sha(stage/'model.pt')))
    print(json.dumps(read(OUT/'delivery.json')),flush=True)


if __name__=='__main__': main()
