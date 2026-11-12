"""串行执行四个预登记候选；保留旧最优模型，失败记录后继续其他候选。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/candidates-v6-20261106'
GROUPS=[('x3d_m',8,2e-4),('tsm_r50',8,1e-4),('mvit_v2_s',4,5e-5),('uniformer_s',4,1e-4)]

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def main():
    OUT.mkdir(exist_ok=True)
    plan=dict(groups=GROUPS,epochs_each=6,seed=42,cache='cache/crop16-192-v3',frames=16,
              architecture_specific='TSM subsamples 8 frames; MViT resizes cached192 to224 inside model; other models16x192',
              head_lr_multiplier=10,validation='466 public clips, evaluation/model selection only',
              training='1995 provided train clips only',extra_modalities=False,test_accessed=False,
              fusion_plan='single view only; each new candidate with old R34 or Swin, new weight .25/.5/.75; also X3D with old ensemble if total<=100M',
              selection='accuracy then fixed40 macroF1; raw-image and offline checks before promotion',
              scope='six epoch initial screening, not final architecture ranking; different initial pretraining from old baseline',
              linux_verified=False)
    if not (OUT/'plan.json').exists():write(OUT/'plan.json',plan)
    results=[]
    for name,batch,lr in GROUPS:
        probe=json.loads((OUT/f'{name}-probe.json').read_text())
        if not probe['passed']:raise RuntimeError('Probe failed: '+name)
        run=ROOT/'runs'/f'{name}-v6-20261106'
        cmd=[sys.executable,'optimized/train_candidates.py','--architecture',name,
             '--initial',f'weights/{name}-k400-v6.pth','--cache','cache/crop16-192-v3','--output',str(run),
             '--epochs','6','--min-epochs','6','--patience','6','--batch',str(batch),'--lr',str(lr),
             '--head-multiplier','10','--seed','42']
        if not (run/'summary.json').exists():
            if (run/'resume.pt').exists():cmd.append('--resume')
            elif run.exists():raise RuntimeError('Incomplete run without checkpoint; inspect: '+str(run))
            write(OUT/'progress.json',dict(stage='training',model=name,command=cmd,time=time.time()))
            with (OUT/f'{name}.log').open('a',encoding='utf-8') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                results.append(dict(name=name,error='training failed',returncode=result.returncode))
                write(OUT/'results-partial.json',results)
                continue
        results.append(dict(name=name,run=str(run),**json.loads((run/'summary.json').read_text())))
        write(OUT/'results-partial.json',results)
        print(json.dumps(results[-1]),flush=True)
    write(OUT/'results.json',results)
    write(OUT/'progress.json',dict(stage='training_complete',time=time.time()))
    if any('error' not in r for r in results):
        subprocess.run([sys.executable,'optimized/finalize_candidates_v6.py'],cwd=ROOT,check=True)

if __name__=='__main__':main()
