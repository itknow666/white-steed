"""检查不同架构是否互补；仅在公开验证集选择预先固定的三个概率平均比例。"""
from collections import OrderedDict
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(run, flip):
    mode = 'flip' if flip else 'single'
    a = np.load(run/f'cached-{mode}-predictions.npz')
    return a['probabilities'], a['labels'], a['sample_ids'].tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r34', type=Path, required=True)
    p.add_argument('--swin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--r34-best-view', action='store_true',help='从已经记录的两个固定视角选择R34视角')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous fusion run')
    config_a, config_b = read(a.r34/'config.json'), read(a.swin/'config.json')
    if not config_a['person_crop'] or not config_b['person_crop'] or config_b['architecture'] != 'swin3dt':
        raise ValueError('Expected shared published crop data and Swin-T')
    view_b = max(read(a.swin/'cached-view-comparison.json'), key=lambda x:(x['accuracy'],x['macro_f1']))
    view_a = max(read(a.r34/'cached-view-comparison.json'), key=lambda x:(x['accuracy'],x['macro_f1'])) if a.r34_best_view else {'flip_tta':True}
    pa, labels, ids = load(a.r34, view_a['flip_tta'])
    pb, labels_b, ids_b = load(a.swin, view_b['flip_tta'])
    order = {name:i for i,name in enumerate(ids_b)}
    if set(ids) != set(ids_b) or len(ids) != 466 or len(order) != 466:
        raise ValueError('Validation sample IDs differ')
    indices = [order[name] for name in ids]
    pb = pb[indices]
    if not np.array_equal(labels,labels_b[indices]):
        raise ValueError('Validation labels differ')
    from train_generalize import metrics
    comparisons = []
    for alpha in [.25,.5,.75]:
        m = metrics(labels,(1-alpha)*pa+alpha*pb)
        comparisons.append({'swin_weight':alpha,'accuracy':m['accuracy'],'macro_f1':m['macro_f1']})
    best = max(comparisons,key=lambda x:(x['accuracy'],x['macro_f1']))
    cache_a, cache_b = read(Path(config_a['cache'])/'manifest.json'),read(Path(config_b['cache'])/'manifest.json')
    count = config_a['total_parameters']+config_b['total_parameters']-cache_a['detector_parameters']
    if count > 100000000 or cache_a['detector_sha256'] != cache_b['detector_sha256']:
        raise ValueError('Parameter limit or detector mismatch')
    a.output.mkdir()
    state = OrderedDict()
    for i,run in enumerate([a.r34,a.swin]):
        for name,value in torch.load(run/'best_ema.pt',map_location='cpu',weights_only=True).items():
            state[f'{i}.{name}']=value
    torch.save(state,a.output/'best_ema.pt')
    components = [{'architecture':'r34','frames':config_a.get('input_frames',cache_a['shape_per_sample'][1]),
                   'size':cache_a['shape_per_sample'][2],'crop_scale':cache_a.get('crop_scale',1.4),
                   'flip_tta':view_a['flip_tta'],'weight':1-best['swin_weight']},
                  {'architecture':'swin3dt','frames':config_b['input_frames'],'size':cache_b['shape_per_sample'][2],
                   'crop_scale':cache_b.get('crop_scale',1.4),'flip_tta':view_b['flip_tta'],'weight':best['swin_weight']}]
    config = {'architecture':'ensemble','components':components,'cache':config_a['cache'],
              'input_frames':components[0]['frames'],'person_crop':True,'total_parameters':count,
              'crop_scale':cache_a.get('crop_scale',1.4),
              'training_source':'optimized/combine_models.py','source_runs':[str(a.r34),str(a.swin)],
              'source_sha256':[sha(a.r34/'best_ema.pt'),sha(a.swin/'best_ema.pt')],
              'selection':'validation only, fixed weights .25/.5/.75; R34 '+('best same-weight view' if a.r34_best_view else 'fixed flip')+', Swin best same-weight view',
              'test_accessed':False,'extra_parameter_updates':0}
    (a.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (a.output/'cached-fusion-comparison.json').write_text(json.dumps({'best':best,'comparisons':comparisons,'parameters':count},indent=2),encoding='utf-8')
    print(json.dumps({'best':best,'parameters':count,'output':str(a.output)}),flush=True)


if __name__ == '__main__':
    main()
