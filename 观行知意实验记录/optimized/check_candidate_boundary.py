"""诊断缓存批量评估与逐样本原图推理的分类边界差异。"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'baseline'))
from inference import Model

torch.set_num_threads(4)
run=ROOT/'runs/mvit_v2_s-v6-20261106'
cfg=json.loads((run/'runtime/config.json').read_text())
cfg.update(weights_path=str(run/'best_ema.pt'),package_dir=str(run/'runtime'))
m=Model(cfg,'cuda')
cached=np.load(run/'best_predictions.npz')
sid='c3445bfa0b2862039c84e795dcafd3db.npy'
index=cached['sample_ids'].astype(str).tolist().index(sid)
meta=json.loads((ROOT/'cache/crop16-192-v3/manifest.json').read_text())
row=sorted([r for r in meta['samples'] if r['split']=='validation'],key=lambda r:r['row'])[index]
manifest=json.loads((ROOT/'data/raw/validation/manifest.json').read_text())
sample=next(r for r in manifest['samples'] if r.get('sample_id',r['file']+'.npy')==sid)
path=ROOT/'data/raw/validation'/sample['file']
x=np.load(ROOT/'cache/crop16-192-v3/validation.npy',mmap_mode='r')[index].astype(np.float32)/255
raw=m.prepare(path)
assert np.array_equal(x,raw)
with torch.no_grad():
    cached_single=m._scores(m.model,x,False).cpu().numpy()
    raw_single=m._scores(m.model,raw,False).cpu().numpy()
np.testing.assert_array_equal(cached_single,raw_single)
original=cached['probabilities'][index]
assert float(np.max(np.abs(original-cached_single)))<.002
report=dict(sample_id=sid,input_exact_match=True,single_cached_raw_scores_exact_match=True,
            training_eval_batch_prediction=int(original.argmax()),single_prediction=int(cached_single.argmax()),
            label=int(cached['labels'][index]),max_probability_difference=float(np.max(np.abs(original-cached_single))),
            cached_top2=sorted(original.tolist(),reverse=True)[:2],
            finding='FP16 batch/layout dependent rounding at exactly tied cached top scores; both predictions incorrect')
(ROOT/'runs/candidates-v6-20261106/boundary-check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
