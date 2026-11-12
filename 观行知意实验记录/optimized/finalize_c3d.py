"""完整图片复验C3D，评估预登记的C3D+X3D融合，并在确认有效时保留最优版本。"""
from collections import OrderedDict
import csv,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from train_c3d import metrics
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/c3d-20261110';X3D=ROOT/'runs/x3d_m-v6-20261106'
FUSION=ROOT/'runs/c3d-x3d-20261110'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def predictions(run):
 a=np.load(run/'best_predictions.npz');ids=a['sample_ids'].astype(str);o=np.argsort(ids)
 assert len(set(ids))==len(ids)==466
 return ids[o],a['labels'][o],a['probabilities'][o]
def evaluate(run,name,check_cache=False):
 package=ROOT/'submission'/name
 if package.exists():raise FileExistsError('Preserve existing package: '+str(package))
 cmd=[sys.executable,'optimized/evaluate_c3d.py','--run',str(run),'--crop','--package-name',name]
 if check_cache:cmd+=['--check-cache',str(ROOT/'cache/crop16-v2')]
 subprocess.run(cmd,cwd=ROOT,check=True)
 r=read(package/'metrics.json');off=read(package/'offline-check.json')
 assert off['passed'] and off['score_count']==40 and off['parameters']<=100_000_000
 assert sha(package/'model.pt')==r['weights_sha256'] and sha(package/'code.zip')==r['code_sha256']
 with zipfile.ZipFile(package/'code.zip') as z:
  assert z.testzip() is None and len(z.infolist())<=256 and sum(i.file_size for i in z.infolist())<=64*2**20
 assert (package/'code.zip').stat().st_size<=20*2**20
 return package,r
def csvmap(path):
 with Path(path).open(encoding='utf-8',newline='') as f:
  rd=csv.DictReader(f);assert rd.fieldnames==['sample_id','predicted_class'];rows=list(rd)
 assert len(rows)==len({r['sample_id'] for r in rows})==466
 return {r['sample_id']:int(r['predicted_class']) for r in rows}
def main():
 summary=read(RUN/'summary.json');ids,labels,p=predictions(RUN)
 single,single_result=evaluate(RUN,'c3d-20261110',True)
 xids,xlabels,xp=predictions(X3D);assert np.array_equal(ids,xids) and np.array_equal(labels,xlabels)
 cconfig=read(RUN/'config.json');xconfig=read(X3D/'config.json')
 detector=read(ROOT/'cache/crop16-v2/manifest.json')['detector_parameters']
 count=cconfig['total_parameters']+xconfig['total_parameters']-detector;assert count<=100_000_000
 comparisons=[]
 for w in [.25,.5,.75]:
  probs=w*p+(1-w)*xp;m=metrics(labels,probs)
  comparisons.append({'c3d_weight':w,'accuracy':m['accuracy'],'macro_f1':m['macro_f1']})
 best=max(comparisons,key=lambda q:(q['accuracy'],q['macro_f1']));w=best['c3d_weight']
 write(RUN/'fusion-comparison.json',{'comparisons':comparisons,'selected':best,'parameters':count})
 FUSION.mkdir(exist_ok=False)
 state=OrderedDict()
 for i,run in enumerate([RUN,X3D]):
  for k,v in torch.load(run/'best_ema.pt',map_location='cpu',weights_only=True).items():state[f'{i}.{k}']=v
 torch.save(state,FUSION/'best_ema.pt')
 components=[{'architecture':'c3d','frames':16,'size':128,'crop_scale':1.4,'flip_tta':False,'weight':w},
             {'architecture':'x3d_m','frames':16,'size':192,'crop_scale':1.4,'flip_tta':False,'weight':1-w}]
 config={'architecture':'ensemble','components':components,'cache':str(ROOT/'cache/crop16-v2'),
         'input_frames':16,'person_crop':True,'crop_scale':1.4,'total_parameters':count,
         'training_source':'optimized/train_c3d.py','source_runs':[str(RUN),str(X3D)],
         'source_sha256':[sha(RUN/'best_ema.pt'),sha(X3D/'best_ema.pt')],
         'selection':'fixed weights 0.25/0.5/0.75; public validation only','test_accessed':False}
 write(FUSION/'config.json',config)
 fused,fusion_result=evaluate(FUSION,'c3d-x3d-20261110')
 previous=read(ROOT/'runs/latest-model.json')
 chosen=max([(RUN,single,single_result),(FUSION,fused,fusion_result)],
            key=lambda q:(q[2]['metrics']['accuracy'],q[2]['metrics']['macro_f1']))
 old=csvmap(ROOT/previous['predictions']);new=csvmap(chosen[1]/'validation.csv');assert set(old)==set(new)==set(ids)
 oldok=np.array([old[k] for k in ids])==labels;newok=np.array([new[k] for k in ids])==labels
 paired={'old_correct':int(oldok.sum()),'new_correct':int(newok.sum()),
         'fixed':int((~oldok&newok).sum()),'lost':int((oldok&~newok).sum())}
 better=(chosen[2]['metrics']['accuracy'],chosen[2]['metrics']['macro_f1'])>(previous['accuracy'],previous['macro_f1'])
 if better:
  run,pkg,res=chosen;code=pkg/'code'
  with zipfile.ZipFile(pkg/'code.zip') as z:z.extractall(code)
  cfg=read(code/'config.json');cfg['weights_file']='../model.pt';write(code/'config.json',cfg)
  latest=dict(run=run.relative_to(ROOT).as_posix(),package=pkg.relative_to(ROOT).as_posix(),
              weights=(pkg/'model.pt').relative_to(ROOT).as_posix(),inference_config=(code/'config.json').relative_to(ROOT).as_posix(),
              predictions=(pkg/'validation.csv').relative_to(ROOT).as_posix(),**res['metrics'],weights_sha256=res['weights_sha256'],smoke=False)
  tmp=ROOT/'runs/latest-model.json.tmp';write(tmp,latest);tmp.replace(ROOT/'runs/latest-model.json')
 outcome={'training':summary,'single':single_result,'fusion':fusion_result,'previous':previous,'paired':paired,
          'promoted':better,'raw_image_validation_samples':466,'offline_python_connection_blocked':True,'linux_verified':False}
 write(RUN/'outcome.json',outcome)
 text=(f'# C3D 训练结果（2026-11-10）\n\n'
       f'单模型：准确率 {single_result["metrics"]["accuracy"]:.2f}%，Macro F1 {single_result["metrics"]["macro_f1"]:.2f}%；'
       f'含检测器 {single_result["parameters"]:,} 个参数。\n\n'
       f'C3D+X3D：准确率 {fusion_result["metrics"]["accuracy"]:.2f}%，Macro F1 {fusion_result["metrics"]["macro_f1"]:.2f}%；'
       f'含共享检测器 {count:,} 个参数。\n\n'
       f'旧模型：准确率 {previous["accuracy"]:.2f}%，Macro F1 {previous["macro_f1"]:.2f}%；更新默认最优版本：{better}。'
       f'纠正 {paired["fixed"]} 段，新增错误 {paired["lost"]} 段。\n\n'
       f'两个模型包均完整读取466段原图并通过Windows Python网络拦截下的离线检查，'
       f'参赛参数和ZIP大小限制检查通过；Linux环境未验证，额外依赖需按赛事要求登记。\n\n'
       f'公开验证集仅用于选择和评估，多次选择会带来乐观偏差；隐藏测试集未知。'
       f'预训练权重源自Sports1M预训练并在UCF101微调，来源和SHA256见competition/c3d-20261110/provenance.json。\n')
 (RUN/'实验结果.md').write_text(text,encoding='utf-8')
 print(json.dumps(outcome,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
