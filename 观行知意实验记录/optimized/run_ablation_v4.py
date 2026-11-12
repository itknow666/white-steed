"""串行执行预先登记的五组实验；失败即停，支持从已完成实验继续。"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/ablation-v4-20261027'
sys.path.insert(0,str(Path(__file__).parent))
from train import write, sha


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def execute(name, cache, frozen=0, lr=1e-5):
    output = ROOT/'runs'/f'{name}-v4-20261027'
    cmd = [sys.executable,'optimized/train_ablation.py','--cache',str(cache),'--output',str(output),
           '--initial','runs/crop16-v2-20261025/best_ema.pt','--epochs','6','--min-epochs','6',
           '--batch','8','--lr',str(lr),'--head-multiplier','5','--patience','6','--seed','42',
           '--freeze-early-epochs',str(frozen),'--unfreeze-lr-factor','0.5']
    if not (output/'summary.json').is_file():
        if (output/'resume.pt').is_file(): cmd.append('--resume')
        elif output.exists():
            raise RuntimeError(f'Incomplete run without epoch checkpoint: {output}. Inspect before recovery.')
        write(OUT/'progress.json',{'stage':'training','run':output.name,'command':cmd,'time':time.time()})
        with (OUT/f'{name}.log').open('a',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    summary = read(output/'summary.json')
    # 视角对照只比较两个固定选项；完整结果保留，最终由真实图像复验。
    if not (output/'cached-view-comparison.json').is_file():
        with (OUT/f'{name}-views.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'optimized/compare_views.py','--run',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = dict(name=name,run=str(output),cache=str(cache),frozen=frozen,lr=lr,**summary)
    print(json.dumps(result),flush=True)
    return result


def score(row): return row['accuracy'],row['macro_f1']


def main():
    OUT.mkdir(exist_ok=True)
    plan = {'epochs_each':6,'seed':42,'initial':'runs/crop16-v2-20261025/best_ema.pt',
            'initial_sha256':sha(ROOT/'runs/crop16-v2-20261025/best_ema.pt'),
            'groups':['control crop1.4 full lr1e-5','wide crop1.8 full lr1e-5',
                      'selected crop staged2+4 lr1e-5 then factor0.5',
                      'selected crop/strategy lr5e-6','selected crop/strategy lr2e-5'],
            'selection':'accuracy then fixed40 macroF1; ties keep simpler control',
            'common':'16frames128, batch8, dropout0.5, smoothing0.05, classpower0.25, BNmomentum0.01, EMA0.99; no mixup/erase',
            'unfreeze':'freeze stem/layer1/layer2 parameters and BN statistics for first2 epochs; all trainable last4; global cosine times0.5 after unfreeze',
            'validation_role':'evaluation and selection only, no gradient/BN/weights training',
            'test_accessed':False,'previous_best_preserved':True,'linux_verified':False}
    if (OUT/'plan.json').exists():
        if read(OUT/'plan.json') != plan: raise ValueError('Registered plan differs')
    else: write(OUT/'plan.json',plan)
    rows=[]
    control = execute('control16',ROOT/'cache/crop16-v2');rows.append(control)
    wide_cache = ROOT/'cache/wide16-v4-20261027'
    # 缓存生成进程独立运行；只接受完成清单，不读取半写入数组。
    deadline=time.monotonic()+1800
    while not (wide_cache/'audit.json').is_file():
        if time.monotonic()>deadline: raise TimeoutError('Wide cache not complete')
        time.sleep(5)
    wide = execute('wide16',wide_cache);rows.append(wide)
    crop_best = max([control,wide],key=score)
    staged = execute('staged16',Path(crop_best['cache']),frozen=2);rows.append(staged)
    strategy = max([crop_best,staged],key=score)
    for name,lr in [('lr5e6',5e-6),('lr2e5',2e-5)]:
        rows.append(execute(name,Path(strategy['cache']),frozen=strategy['frozen'],lr=lr))
        write(OUT/'results-partial.json',rows)
    best = max(rows,key=score)
    write(OUT/'results.json',{'groups':rows,'best_single_view':best,
                            'note':'Compare cached views and existing best fusion separately before promotion.',
                            'epochs_total':sum(r['epochs_completed'] for r in rows),'test_accessed':False})
    write(OUT/'progress.json',{'stage':'training_complete','best':best,'time':time.time()})


if __name__=='__main__':
    try: main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        write(OUT/'failure.json',{'error':repr(error),'time':time.time()})
        raise
