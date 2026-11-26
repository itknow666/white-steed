"""Compare fixed view and backbone combinations on published validation clips."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

run = Path(__file__).resolve().parent
root = run.parents[1]
baseline = root / 'submission/v8-input-resolution-20261123/crop192-control'
cnn_path = root / 'runs/x3d_m-v6-20261106/best_ema.pt'
sys.path.insert(0, str(baseline / 'code'))
from inference import Model

# Each coefficient list applies to the same ordered features below.
plans = {
    'baseline': [1., 0., 0., 0.],
    'two-timescales': [.65, .35, 0., 0.],
    'transformer-plus-cnn': [.8, 0., 0., .2],
    'wider-context': [.65, 0., .35, 0.],
    'joint': [.5, .2, .15, .15],
}
config = json.loads((baseline / 'code/config.json').read_text(encoding='utf-8'))
config.update(weights_path=str(baseline / 'model.pt'), package_dir=str(baseline / 'code'))
reference = Model(config, 'cuda')
cnn = Model._classifier('x3d_m')
cnn.load_state_dict(torch.load(cnn_path, map_location='cpu', weights_only=True), strict=True)
cnn.cuda().eval()
combined_parameters = reference.parameters + sum(p.numel() for p in cnn.parameters())
assert combined_parameters <= 100_000_000

def digest(path):
    with path.open('rb') as handle:
        sha = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()

(run / 'design.json').write_text(json.dumps({
    'components_order': ['16f-192-standard-crop', '32f-192-standard-crop',
                         '16f-192-wide-crop', 'X3D-M-16f-192-standard-crop'],
    'plans': plans, 'crop_scales': [1.4, 1.65],
    'main_sha256': digest(baseline/'model.pt'), 'cnn_sha256': digest(cnn_path),
    'cnn_run': 'x3d_m-v6-20261106', 'cnn_pretrain': 'Kinetics-400',
    'main_parameters': reference.parameters, 'combined_parameters': combined_parameters,
}, indent=2), encoding='utf-8')

dataset = root / 'data/raw/validation'
manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
ids, labels, predictions, timings = [], [], [], []
phase = dict(config, frames=32)

@torch.inference_mode()
def score_clip(path):
    start = time.perf_counter()
    depth, ir, crops = reference._paths_crop(path, all_scales=True)
    detection_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    standard = reference._input(depth, ir, crops[1.4], config)
    main = reference._scores(reference.model, standard, False)
    main_end = time.perf_counter()
    # Fewer than 24 available frames leaves too few extra temporal observations.
    dense = main
    if len(ir) >= 24:
        # Resize temporal positions independently for each of the 14x14 spatial patches.
        encoder = reference.model.encoder
        original_positions = encoder.pos_embed
        spatial_tokens = 14 * 14
        positions = original_positions.reshape(1, 8, spatial_tokens * original_positions.shape[-1])
        resized = torch.nn.functional.interpolate(
            positions.transpose(1, 2), size=16, mode='linear', align_corners=True)
        encoder.pos_embed = resized.transpose(1, 2).reshape(1, 16 * spatial_tokens, -1)
        try:
            dense = reference._scores(reference.model, reference._input(depth, ir, crops[1.4], phase), False)
        finally:
            encoder.pos_embed = original_positions
    dense_end = time.perf_counter()
    wider = reference._scores(reference.model, reference._input(depth, ir, crops[1.65], config), False)
    wide_end = time.perf_counter()
    convolution = reference._scores(cnn, standard, False)
    cnn_end = time.perf_counter()
    return torch.stack((main, dense, wider, convolution)).cpu().numpy(), [
        detection_ms, 1000*(main_end-start), 1000*(dense_end-main_end),
        1000*(wide_end-dense_end), 1000*(cnn_end-wide_end),
    ]

# Expose the second crop scale to the existing detector without loading a second detector.
reference.components = [{'crop_scale': 1.65}]
for index, sample in enumerate(manifest['samples'], 1):
    planes, timing = score_clip(dataset / sample['file'])
    predictions.append(planes)
    timings.append(timing)
    labels.append(int(sample['label']))
    ids.append(sample.get('sample_id', sample['file']+'.npy'))
    if index % 50 == 0 or index == 466:
        print(f'Scored {index}/466', flush=True)
assert len(ids) == len(set(ids)) == 466
prob = np.stack(predictions)
truth = np.asarray(labels)
np.savez_compressed(run/'component-scores.npz', sample_ids=ids, probabilities=prob)
control = prob[:,0].argmax(1)
with (baseline/'validation.csv').open(newline='',encoding='utf-8') as f:
    from_file = {row['sample_id']:int(row['predicted_class']) for row in csv.DictReader(f)}
assert dict(zip(ids, control)) == from_file, 'Baseline replay differs from saved canonical predictions'

results = {}
for name, weights in plans.items():
    pred = (prob * np.asarray(weights, dtype=np.float32)[None,:,None]).sum(1).argmax(1)
    cm = np.bincount(40 * truth + pred, minlength=1600).reshape(40,40)
    scores = {'correct': int(cm.trace()),
              'macro_f1': float((2*cm.diagonal()/np.maximum(1,cm.sum(0)+cm.sum(1))).mean()),
              'fixed': int(((pred==truth)&(control!=truth)).sum()),
              'regressed': int(((pred!=truth)&(control==truth)).sum())}
    results[name] = scores
    with (run/f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id','predicted_class'])
        writer.writerows(zip(ids,pred))
result = {'arms': results, 'mean_ms_detection_and_components': np.asarray(timings[1:]).mean(0).tolist()}
(run/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result), flush=True)
