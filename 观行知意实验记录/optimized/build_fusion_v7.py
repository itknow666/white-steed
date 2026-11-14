"""v7：把 VideoMAE-S(0.75) 与 R34 control16(0.25) 组合为 ensemble 权重与配置；不做任何梯度更新。"""
from collections import OrderedDict
from pathlib import Path
import hashlib, json
import torch

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
vm = ROOT/'runs/videomae2_s-v7-20261114'
r34 = ROOT/'runs/control16-v4-20261027'
out = ROOT/'runs/fusion-videomae2-r34-v7-20261114'
if out.exists():
    raise FileExistsError(out)
state = OrderedDict()
for i, run in enumerate([vm, r34]):
    for k, v in torch.load(run/'best_ema.pt', map_location='cpu', weights_only=True).items():
        state[f'{i}.{k}'] = v
cv = json.loads((vm/'config.json').read_text(encoding='utf-8'))
cr = json.loads((r34/'config.json').read_text(encoding='utf-8'))
det = 2624080
params = (cv['total_parameters']-det) + (cr['total_parameters']-det) + det
if params > 100_000_000:
    raise ValueError('over 100M')
out.mkdir()
torch.save(state, out/'best_ema.pt')
config = {'architecture': 'ensemble',
          'components': [
              {'architecture': 'videomae2_s', 'frames': 16, 'size': 192, 'crop_scale': 1.4, 'flip_tta': False, 'weight': 0.75},
              {'architecture': 'r34', 'frames': 16, 'size': 128, 'crop_scale': 1.4, 'flip_tta': False, 'weight': 0.25}],
          'cache': cv['cache'], 'input_frames': 16, 'person_crop': True, 'total_parameters': params, 'crop_scale': 1.4,
          'training_source': 'optimized/build_fusion_v7.py',
          'source_runs': ['runs/videomae2_s-v7-20261114', 'runs/control16-v4-20261027'],
          'source_sha256': [sha(vm/'best_ema.pt'), sha(r34/'best_ema.pt')],
          'selection': 'validation only, fixed grid .25/.5/.75 for VideoMAE weight (fusion-v7-cached.json)',
          'test_accessed': False, 'extra_parameter_updates': 0}
(out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'parameters': params, 'keys': len(state)}))