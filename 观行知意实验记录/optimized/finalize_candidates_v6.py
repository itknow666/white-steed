"""比较已登记的候选与融合；从原始图片复验胜者，离线通过后才更新最优指针。"""
from collections import OrderedDict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import torch
from train_candidates import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/candidates-v6-20261106'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(r):return r['accuracy'],r['macro_f1']

def predictions(run,old=False):
    a=np.load(run/('cached-single-predictions.npz' if old else 'best_predictions.npz'))
    ids=a['sample_ids'].astype(str)
    order=np.argsort(ids)
    assert len(ids)==len(set(ids))==466
    return ids[order],a['labels'][order],a['probabilities'][order]

def component(run,weight):
    c=read(run/'config.json');meta=read(Path(c['cache'])/'manifest.json')
    return dict(architecture=c.get('architecture','r34'),frames=c['input_frames'],
                size=meta['shape_per_sample'][2],crop_scale=meta.get('crop_scale',1.4),
                flip_tta=False,weight=weight)

def validate(run,package,ids,probabilities):
    if package.exists():
        if not (package/'metrics.json').exists() or not (package/'offline-check.json').exists():
            raise RuntimeError('Incomplete package requires inspection: '+str(package))
        prior=read(package/'metrics.json')
        assert Path(prior['run']).resolve()==run.resolve()
        assert sha(run/'best_ema.pt')==prior['weights_sha256']
    else:
        subprocess.run([sys.executable,'optimized/evaluate_candidates.py','--run',str(run),'--crop',
                        '--package-name',package.name],cwd=ROOT,check=True)
    with (package/'validation.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['sample_id','predicted_class'];rows=list(reader)
    assert len(rows)==len({r['sample_id'] for r in rows})==466
    expected=dict(zip(ids,probabilities.argmax(1)))
    assert set(expected)=={r['sample_id'] for r in rows}
    differences=[dict(sample_id=r['sample_id'],cached=int(expected[r['sample_id']]),actual=int(r['predicted_class']))
                 for r in rows if expected[r['sample_id']]!=int(r['predicted_class'])]
    if differences:
        # 仅接受已经逐像素、逐分支验证过的指定舍入边界；其他差异仍停止检查。
        boundary_files={'mvit_v2_s-v6-20261106':'boundary-check.json',
                        'fusion-candidates-v6-20261106':'fusion-boundary-check.json'}
        assert run.name in boundary_files and len(differences)==1
        boundary=read(OUT/boundary_files[run.name])
        d=differences[0]
        assert d['sample_id']==boundary['sample_id']
        assert d['cached']==boundary['training_eval_batch_prediction'] and d['actual']==boundary['single_prediction']
        assert boundary['input_exact_match'] and boundary['single_cached_raw_scores_exact_match']
        assert boundary['max_probability_difference']<.002
    assert read(package/'offline-check.json')['passed']
    with zipfile.ZipFile(package/'code.zip') as z:assert z.testzip() is None
    result=read(package/'metrics.json')
    assert sha(package/'model.pt')==result['weights_sha256']
    result['cached_prediction_differences']=differences
    write(package/'validation-reconciliation.json',dict(rows=466,differences=differences,
          authoritative_metrics='actual raw-image single-sample inference',
          explanation='See experiment boundary diagnostics: confirmed same input and same single-sample scores; FP16 batch/layout rounding' if differences else 'All categories match'))
    return result

def main():
    results=[r for r in read(OUT/'results.json') if 'error' not in r]
    best=max(results,key=score)
    single=Path(best['run']);ids,labels,probs=predictions(single)
    single_package=ROOT/'submission/candidates-v6-single-20261106'
    single_result=validate(single,single_package,ids,probs)
    old_r34=ROOT/'runs/control16-v4-20261027';old_swin=ROOT/'runs/swin192-v3-20261025'
    old_models=[old_r34,old_swin]
    old_preds=[predictions(r,True) for r in old_models]
    detector=read(ROOT/'cache/crop16-192-v3/manifest.json')['detector_parameters']
    comparisons=[]
    best_trial=None;best_probs=None
    for result in results:
        run=Path(result['run']);new_ids,new_labels,new_probs=predictions(run)
        assert np.array_equal(new_ids,ids) and np.array_equal(new_labels,labels)
        for j,old_run in enumerate(old_models):
            oi,ol,op=old_preds[j];assert np.array_equal(oi,ids) and np.array_equal(ol,labels)
            count=result['parameters']+read(old_run/'config.json')['total_parameters']-detector
            if count>100000000:continue
            for weight in [.25,.5,.75]:
                p=weight*new_probs+(1-weight)*op;m=metrics(labels,p)
                trial=dict(name=result['name'],sources=[str(run),str(old_run)],weights=[weight,1-weight],
                           parameters=count,accuracy=m['accuracy'],macro_f1=m['macro_f1'])
                comparisons.append(trial)
                if best_trial is None or score(trial)>score(best_trial):best_trial,best_probs=trial,p
        if result['name']=='x3d_m':
            count=result['parameters']+sum(read(r/'config.json')['total_parameters']-detector for r in old_models)
            if count<=100000000:
                old_p=(old_preds[0][2]+old_preds[1][2])/2
                for weight in [.25,.5,.75]:
                    p=weight*new_probs+(1-weight)*old_p;m=metrics(labels,p)
                    trial=dict(name=result['name'],sources=[str(run),*map(str,old_models)],
                               weights=[weight,(1-weight)/2,(1-weight)/2],parameters=count,
                               accuracy=m['accuracy'],macro_f1=m['macro_f1'])
                    comparisons.append(trial)
                    if best_trial is None or score(trial)>score(best_trial):best_trial,best_probs=trial,p
    write(OUT/'fusion-comparisons.json',dict(best=best_trial,trials=comparisons))
    fusion=ROOT/'runs/fusion-candidates-v6-20261106'
    fusion_exists=fusion.exists()
    if not fusion_exists:fusion.mkdir()
    state=OrderedDict()
    for i,source in enumerate(best_trial['sources']):
        for k,v in torch.load(Path(source)/'best_ema.pt',map_location='cpu',weights_only=True).items():state[f'{i}.{k}']=v
    if fusion_exists:
        prior_state=torch.load(fusion/'best_ema.pt',map_location='cpu',weights_only=True)
        assert state.keys()==prior_state.keys()
        assert all(torch.equal(v,prior_state[k]) for k,v in state.items())
        del prior_state
    else:torch.save(state,fusion/'best_ema.pt')
    config=dict(architecture='ensemble',components=[component(Path(s),w) for s,w in zip(best_trial['sources'],best_trial['weights'])],
                cache=str(ROOT/'cache/crop16-192-v3'),input_frames=16,person_crop=True,crop_scale=1.4,
                total_parameters=best_trial['parameters'],training_source='optimized/train_candidates.py',
                source_runs=best_trial['sources'],source_sha256=[sha(Path(s)/'best_ema.pt') for s in best_trial['sources']],
                selection='preregistered fixed fusion weights; public validation only',test_accessed=False)
    if fusion_exists:assert read(fusion/'config.json')==config
    else:write(fusion/'config.json',config)
    fusion_package=ROOT/'submission/candidates-v6-fusion-20261106'
    fusion_result=validate(fusion,fusion_package,ids,best_probs)
    options=[(single,single_package,single_result),(fusion,fusion_package,fusion_result)]
    selected=max(options,key=lambda r:score(r[2]['metrics']))
    run,package,result=selected;previous=read(ROOT/'runs/latest-model.json')
    promoted=score(result['metrics'])>score(previous)
    def csv_map(path):
        with path.open(encoding='utf-8',newline='') as f:
            return {r['sample_id']:int(r['predicted_class']) for r in csv.DictReader(f)}
    old_map=csv_map(ROOT/previous['predictions']);new_map=csv_map(package/'validation.csv')
    assert set(old_map)==set(new_map)==set(ids)
    old_correct=np.array([old_map[k] for k in ids])==labels
    new_correct=np.array([new_map[k] for k in ids])==labels
    paired=dict(old_correct=int(old_correct.sum()),new_correct=int(new_correct.sum()),
                fixed=int((~old_correct&new_correct).sum()),lost=int((old_correct&~new_correct).sum()))
    write(OUT/'paired-comparison.json',paired)
    if promoted:
        code=package/'code'
        with zipfile.ZipFile(package/'code.zip') as z:z.extractall(code)
        cfg=read(code/'config.json');cfg['weights_file']='../model.pt';write(code/'config.json',cfg)
        latest=dict(run=run.relative_to(ROOT).as_posix(),package=package.relative_to(ROOT).as_posix(),
                    weights=(package/'model.pt').relative_to(ROOT).as_posix(),
                    inference_config=(code/'config.json').relative_to(ROOT).as_posix(),
                    predictions=(package/'validation.csv').relative_to(ROOT).as_posix(),
                    **result['metrics'],weights_sha256=result['weights_sha256'],smoke=False)
        temp=ROOT/'runs/latest-model.json.tmp';write(temp,latest);temp.replace(ROOT/'runs/latest-model.json')
    outcome=dict(single=single_result,fusion=fusion_result,previous=previous,promoted=promoted,
                 paired=paired,actual_cached_match=not(single_result['cached_prediction_differences'] or fusion_result['cached_prediction_differences']),
                 offline_passed=True,linux_verified=False)
    write(OUT/'outcome.json',outcome)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for r in results:
        history=[json.loads(line) for line in (Path(r['run'])/'history.jsonl').read_text().splitlines()]
        for ax,key,title in zip(axes,['accuracy','macro_f1'],['Validation accuracy (%)','Fixed-40 Macro F1 (%)']):
            ax.plot([h['epoch'] for h in history],[h[key] for h in history],marker='o',label=r['name'])
            ax.set_xlabel('Epoch');ax.set_ylabel(title);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'learning-curves.png',dpi=160);plt.close(fig)
    lines=['# 四种公开视频模型首轮筛选（2026-11-06）','','每个模型使用1995段训练数据微调6轮；466段公开验证集仅用于评估和模型选择。',
           '这是短训结果，不是架构上限；公开验证集被反复选择，隐藏测试表现未知。','',
           '|模型|准确率 %|Macro F1 %|最佳轮次|','|---|---:|---:|---:|']
    for r in results:lines.append(f'|{r["name"]}|{r["accuracy"]:.2f}|{r["macro_f1"]:.2f}|{r["best_epoch"]}|')
    lines += ['',f'最佳融合：{fusion_result["metrics"]}；更新最优模型：{promoted}。',
              f'对比旧版：新模型纠正 {paired["fixed"]} 段，也损失 {paired["lost"]} 段。',
              '最佳单模型与融合模型均经过全部466段原始图片推理复验；离线Python网络拦截加载测试通过。',
              f'单模型实际推理指标：{single_result["metrics"]}。缓存/实际类别差异数：单模型 {len(single_result["cached_prediction_differences"])}、融合 {len(fusion_result["cached_prediction_differences"])}。',
              'MViT有1个FP16并列最高分边界：批量缓存预测6、逐样本预测7，正确标签8；输入逐像素相同，逐样本缓存与原图输出一致。准确率不变，实际Macro F1略有变化。见boundary-check.json；最终以实际推理结果为准。',
              '融合也有1个FP16近边界样本：批量缓存预测37、逐样本预测1，正确标签19；三个分支逐像素输入及逐样本输出一致。见fusion-boundary-check.json。两种结果均错误，不影响准确率；采用实际Macro F1。',
              'Linux未验证。新增依赖需按比赛要求登记。',
              '输入：同步16帧192缓存；TSM内部取8帧，MViT内部上采样224，不能据此宣称完全等计算量比较。',
              '本轮从通用Kinetics400预训练开始；旧R34有不同的预训练来源和更多本题微调历史，短训分数不能直接等同于架构优劣。',
              '来源与SHA256：competition/candidates-v6/weights-provenance.json。许可证保存在同目录。',
              '复现入口：optimized/run_candidates_v6.py，单个训练入口：optimized/train_candidates.py。',
              f'单模型包：{single_package}',f'融合包：{fusion_package}']
    (OUT/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'progress.json',dict(stage='complete',promoted=promoted))
    print(json.dumps(outcome,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
