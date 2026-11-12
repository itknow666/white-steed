"""对本轮融合中的已发现边界样本，逐分支证明原图与缓存预处理一致。"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'baseline'))
from inference import Model
torch.set_num_threads(4)
run=ROOT/'runs/fusion-candidates-v6-20261106'
cfg=json.loads((run/'runtime/config.json').read_text())
cfg.update(weights_path=str(run/'best_ema.pt'),package_dir=str(run/'runtime'))
m=Model(cfg,'cuda');source=json.loads((run/'config.json').read_text())
sid='93bb8a03e08e164fb483da993e6a9b2d.npy'
manifest=json.loads((ROOT/'data/raw/validation/manifest.json').read_text())
sample=next(r for r in manifest['samples'] if r.get('sample_id',r['file']+'.npy')==sid)
path=ROOT/'data/raw/validation'/sample['file']
depth,ir,crops=m._paths_crop(path,all_scales=True)
parts=[];old_parts=[];checks=[]
for net,spec,src in zip(m.model,m.components,source['source_runs']):
    src=Path(src);train=json.loads((src/'config.json').read_text());cache=Path(train['cache'])
    meta=json.loads((cache/'manifest.json').read_text())
    row=next(r for r in meta['samples'] if r['split']=='validation' and r['sample_id']==sid)
    x=np.load(cache/'validation.npy',mmap_mode='r')[row['row']].astype(np.float32)/255
    raw=m._input(depth,ir,crops[spec['crop_scale']],spec)
    assert np.array_equal(x,raw)
    with torch.no_grad():
        a=m._scores(net,x,False).cpu().numpy();b=m._scores(net,raw,False).cpu().numpy()
    np.testing.assert_array_equal(a,b)
    old=np.load(src/('cached-single-predictions.npz' if spec['architecture'] in ['r34','swin3dt'] else 'best_predictions.npz'))
    idx=old['sample_ids'].astype(str).tolist().index(sid);p=old['probabilities'][idx]
    delta=float(np.max(np.abs(a-p)));assert delta<.002
    checks.append(dict(architecture=spec['architecture'],input_exact_match=True,
                       single_cached_raw_scores_exact_match=True,max_probability_difference=delta))
    parts.append(a*spec['weight']);old_parts.append(p*spec['weight'])
original=sum(old_parts);actual=sum(parts)
report=dict(sample_id=sid,training_eval_batch_prediction=int(original.argmax()),single_prediction=int(actual.argmax()),
            label=int(old['labels'][idx]),input_exact_match=True,single_cached_raw_scores_exact_match=True,
            max_probability_difference=float(np.max(np.abs(original-actual))),components=checks,
            cached_top2=sorted(original.tolist(),reverse=True)[:2],
            finding='FP16 batch/layout rounding near decision boundary; both predictions incorrect')
assert report['training_eval_batch_prediction']==37 and report['single_prediction']==1 and report['label']==19
(ROOT/'runs/candidates-v6-20261106/fusion-boundary-check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
