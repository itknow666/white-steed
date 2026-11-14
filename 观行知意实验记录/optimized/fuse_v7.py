"""v7 缓存级融合检查：只读取已保存的验证概率，按预先固定的权重网格平均，不更新任何参数、不改 latest-model.json。"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    a = np.load(path)
    return a['probabilities'], a['labels'], [str(s) for s in a['sample_ids']]

def metrics(labels, p):
    cm = np.bincount(labels*40 + p.argmax(1), minlength=1600).reshape(40, 40)
    tp = cm.diagonal()
    return {'accuracy': float(100*tp.sum()/len(labels)), 'macro_f1': float(100*np.mean(2*tp/np.maximum(1, cm.sum(0)+cm.sum(1))))}

new_run = Path(sys.argv[1])
sources = {
    'videomae2_s': new_run/'best_predictions.npz',
    'r34': ROOT/'runs/control16-v4-20261027/cached-single-predictions.npz',
    'swin3dt': ROOT/'runs/swin192-v3-20261025/cached-single-predictions.npz',
    'x3d_m': ROOT/'runs/x3d_m-v6-20261106/best_predictions.npz',
}
params = {'videomae2_s': 22091944, 'r34': 63514700, 'swin3dt': 27884302, 'x3d_m': 3056850}
DET = 2624080
base_p, labels, ids = load(sources['videomae2_s'])
probs = {'videomae2_s': base_p}
for k, f in sources.items():
    if k == 'videomae2_s': continue
    p, l, i = load(f)
    order = {s: n for n, s in enumerate(i)}
    idx = [order[s] for s in ids]
    if not np.array_equal(l[idx], labels): raise ValueError('label mismatch '+k)
    probs[k] = p[idx]
out = {'single': {k: metrics(labels, v) for k, v in probs.items()}, 'pairs': [], 'triples': []}
for other in ['r34', 'swin3dt', 'x3d_m']:
    for w in [.25, .5, .75]:
        m = metrics(labels, w*probs['videomae2_s'] + (1-w)*probs[other])
        out['pairs'].append(dict(m, components=['videomae2_s', other], videomae_weight=w,
                                 parameters=params['videomae2_s']+params[other]+DET))
for w in [(.375, .375, .25), (.5, .25, .25), (.25, .5, .25)]:
    m = metrics(labels, w[0]*probs['videomae2_s'] + w[1]*probs['r34'] + w[2]*probs['x3d_m'])
    out['triples'].append(dict(m, components=['videomae2_s', 'r34', 'x3d_m'], weights=w,
                               parameters=params['videomae2_s']+params['r34']+params['x3d_m']+DET))
out['note'] = 'cache-level only; public validation used for selection; not verified on raw images; params of swin/x3d from earlier run totals minus detector'
(new_run/'fusion-v7-cached.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))