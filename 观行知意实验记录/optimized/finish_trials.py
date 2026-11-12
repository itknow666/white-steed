"""训练结束后的顺序验证：缓存比较、真实图片复验、离线检查、整理交付。"""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['frozenbn16-v3-20261025','temporal16-v3-20261025','detail192-v3-20261025','swin192-v3-20261025','ircontrast16-v3-20261025']


def call(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True,env=dict(os.environ,PYTHONUTF8='1',YOLO_AUTOINSTALL='false'))


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    call('optimized/train_ir.py','--cache','cache/ircontrast16-v3','--output','runs/ircontrast16-v3-20261025',
         '--initial','runs/crop16-v2-20261025/best_ema.pt','--epochs','6','--batch','8','--lr','0.00001',
         '--head-multiplier','5','--patience','3','--min-epochs','4')
    call('optimized/compare_views.py','--run','runs/ircontrast16-v3-20261025')
    call('optimized/compare_views.py','--run','runs/swin192-v3-20261025')
    call('optimized/compare_views.py','--run','runs/crop32-v2-20261025')
    call('optimized/combine_models.py','--r34','runs/crop32-v2-20261025','--swin','runs/swin192-v3-20261025','--output','runs/fusion-v3-20261025')
    views = [v for name in RUNS for v in read(Path('runs')/name/'cached-view-comparison.json')]
    best = max(views,key=lambda x:(x['accuracy'],x['macro_f1']))
    run = Path(best['run'])
    config = read(run/'config.json')
    cache = config.get('val_cache') or config['cache']
    mode = 'flip' if best['flip_tta'] else 'single'
    args = ['optimized/evaluate.py','--run',run,'--package-name',run.name.replace('-v3-',f'-{mode}-v3-'),'--crop','--check-cache',cache]
    if best['flip_tta']:
        args.append('--flip')
    call(*args)
    fusion = read(Path('runs/fusion-v3-20261025/cached-fusion-comparison.json'))
    baseline = read(Path('submission/optimized-20261025/metrics.json'))
    if fusion['best']['accuracy'] >= baseline['metrics']['accuracy']-.5:
        call('optimized/evaluate.py','--run','runs/fusion-v3-20261025','--package-name','fusion-v3-20261025','--crop','--check-cache','cache/crop32-v2')
    call('optimized/finalize_generalize.py')


if __name__ == '__main__':
    main()
