"""Recalibrate the frozen V8 classifier using official training clips."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_videomae import ROOT, Clips, metrics, save, sha, write
from videomae_v8 import DomainVideoMAE


TRAIN_MANIFEST = ROOT / 'data/raw/train/manifest.json'
CACHE = ROOT / 'cache/crop16-192-v3'
BASE = ROOT / 'submission/v8-teacher-transfer-20261122/kd-non-target'
WEIGHTS = BASE / 'model.pt'
VIEWS = ('clean', 'mirror', 'ir_photometric')
# alpha anchors the newly fitted head. beta penalizes IR-dependent logit changes.
RECIPES = {
    'fresh-linear': {'anchor': False, 'mirror': False, 'alpha': .15, 'beta': 0.},
    'anchored-clean': {'anchor': True, 'mirror': False, 'alpha': 2., 'beta': 0.},
    'anchored-mirror': {'anchor': True, 'mirror': True, 'alpha': 2., 'beta': 0.},
    'anchored-ir-consistency': {'anchor': True, 'mirror': True, 'alpha': 2., 'beta': .15},
    'anchored-ir-normalized': {'anchor': True, 'mirror': True, 'alpha': 2., 'beta': .05, 'normalize_ir': True},
}
COLUMNS = ('从零线性头', '锚定线性头', '锚定加镜像', '微弱 IR 一致性', '归一化 IR 一致性')


def official_split(split, expected):
    clips = Clips(CACHE, split, frames=16)
    manifest_path = ROOT / 'data/raw' / split / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    truth = {row.get('sample_id', row['file']+'.npy'): int(row['label']) for row in manifest['samples']}
    ids = [row['sample_id'] for row in clips.rows]
    labels = np.array([int(row['label']) for row in clips.rows], dtype=np.int64)
    if len(ids) != expected or len(truth) != expected or len(set(ids)) != expected:
        raise ValueError(f'Expected {expected} distinct official {split} clips')
    if set(ids) != truth.keys() or not np.array_equal(labels, [truth[key] for key in ids]):
        raise ValueError(f'{split} cache and release manifest have different samples or labels')
    return clips, ids, labels


@torch.inference_mode()
def extract(output, source_hash):
    clips, ids, labels = official_split('train', 1995)
    model = DomainVideoMAE()
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model.cuda().to(memory_format=torch.channels_last_3d).eval()
    dim = model.encoder.num_features
    features = np.empty((len(ids), len(VIEWS), dim), dtype=np.float32)
    for batch, (x, _, index) in enumerate(DataLoader(clips, batch_size=8, pin_memory=True)):
        x = x.cuda(non_blocking=True)
        augmented = x.clone()
        # Apply a modest, temporally constant exposure shift to IR only.
        gain = torch.where(index.cuda() % 2 == 0, .8, 1.2)[:, None, None, None, None]
        ir = augmented[:, 3:4]
        ir.copy_(((ir - .5) * gain + .5 + (gain - 1) * .15).clamp(0, 1))
        for view_idx, view in enumerate((x, x.flip(-1), augmented)):
            view = view.contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.float16):
                embeddings = model.encoder(model.normalized_clip(view))
                if batch == 0 and view_idx == 0:
                    expected = model(view)
                    torch.testing.assert_close(model.head(embeddings), expected, rtol=0, atol=0)
            features[index.numpy(), view_idx] = embeddings.float().cpu().numpy()
        if batch % 50 == 0:
            print(f'Extracted {int(index[-1])+1}/{len(ids)} training clips', flush=True)
    if not np.isfinite(features).all():
        raise ValueError('Feature cache has nonfinite values')
    temporary = output / 'train-features.partial'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, ids=ids, labels=labels, features=features, views=VIEWS)
    target = output / 'train-features.npz'
    temporary.replace(target)
    write(output / 'feature-source.json', {
        'split': 'train', 'views': list(VIEWS), 'samples': len(ids), 'dimension': dim,
        'source_weights_sha256': source_hash, 'feature_cache_sha256': sha(target),
        'train_manifest_sha256': sha(TRAIN_MANIFEST), 'cache_manifest_sha256': sha(CACHE/'manifest.json'),
    })



def fit(features, labels, state, recipe):
    anchor_weight = state['head.1.weight'].double()
    anchor_bias = state['head.1.bias'].double()
    weight = torch.nn.Parameter(anchor_weight.clone() if recipe['anchor'] else torch.zeros_like(anchor_weight))
    bias = torch.nn.Parameter(anchor_bias.clone() if recipe['anchor'] else torch.zeros_like(anchor_bias))
    prior_weight, prior_bias = weight.detach().clone(), bias.detach().clone()
    clean, mirror, ir = features.unbind(dim=1)
    views = torch.cat((clean, mirror)) if recipe['mirror'] else clean
    targets = labels.repeat(2) if recipe['mirror'] else labels
    class_weights = labels.bincount(minlength=40).double().pow(-.18)
    class_weights /= class_weights.mean()
    ir_delta = ir - clean
    optimizer = torch.optim.LBFGS((weight, bias), max_iter=120, line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-8, tolerance_change=1e-10)
    calls = 0

    def ir_sensitivity(candidate):
        delta = F.linear(ir_delta, candidate)
        # A common shift of all 40 logits leaves the predicted distribution unchanged.
        return (delta - delta.mean(1, keepdim=True)).square().mean()

    baseline_ir_variance = ir_sensitivity(anchor_weight)
    ir_scale = float(baseline_ir_variance.clamp_min(1e-6)) if recipe.get('normalize_ir') else 1.

    def objective():
        classification = F.cross_entropy(F.linear(views, weight, bias), targets,
                                         weight=class_weights, label_smoothing=.05)
        regularization = (weight - prior_weight).square().sum(1).mean()
        regularization = regularization + (bias - prior_bias).square().mean()
        return classification + recipe['alpha'] * regularization + recipe['beta'] / ir_scale * ir_sensitivity(weight)

    def closure():
        nonlocal calls
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        calls += 1
        return loss

    start = time.perf_counter()
    initial_loss = float(optimizer.step(closure).detach())
    elapsed = time.perf_counter() - start
    state_out = dict(state)
    state_out['head.1.weight'] = weight.detach().to(state['head.1.weight'].dtype)
    state_out['head.1.bias'] = bias.detach().to(state['head.1.bias'].dtype)
    with torch.no_grad():
        logits = F.linear(clean, weight, bias)
        stats = {'objective_before': initial_loss, 'objective_after': float(objective()),
                 'training_accuracy': float((logits.argmax(1) == labels).double().mean() * 100),
                 'ir_logit_shift_rms_before': float(baseline_ir_variance.sqrt()),
                 'ir_logit_shift_rms_after': float(ir_sensitivity(weight).sqrt()),
                 'head_weight_update_l2': float((weight - anchor_weight).norm()),
                 'optimizer_evaluations': calls, 'fit_seconds': elapsed}
    if not np.isfinite(list(stats.values())).all() or stats['objective_after'] > initial_loss:
        raise ValueError('Invalid classifier fit')
    return state_out, stats


def make_candidates(output, source_hash):
    cache_file = output / 'train-features.npz'
    metadata = json.loads((output / 'feature-source.json').read_text(encoding='utf-8'))
    if (metadata['source_weights_sha256'] != source_hash or metadata['feature_cache_sha256'] != sha(cache_file)
            or metadata['train_manifest_sha256'] != sha(TRAIN_MANIFEST)
            or metadata['cache_manifest_sha256'] != sha(CACHE/'manifest.json')):
        raise ValueError('Frozen feature provenance differs from the requested source')
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    with np.load(cache_file, allow_pickle=False) as cache:
        labels = torch.from_numpy(cache['labels'])
        features = torch.from_numpy(cache['features']).double()
        if list(cache['views']) != list(VIEWS) or features.shape != (1995, len(VIEWS), state['head.1.weight'].size(1)):
            raise ValueError('Unexpected feature views or shape')
    for name, recipe in RECIPES.items():
        folder = output / name
        folder.mkdir(exist_ok=True)
        if (folder / 'summary.json').is_file():
            continue
        candidate, stats = fit(features, labels, state, recipe)
        save(folder / 'best_ema.pt', candidate)
        write(folder / 'config.json', dict(recipe, source_sha256=source_hash, train_samples=len(labels)))
        write(folder / 'summary.json', dict(stats, mode=name, best_epoch=None))
        print(name, stats, flush=True)


@torch.inference_mode()
def probe_ir(output):
    from sensor_corruption_v8 import stress_clip

    cases = ('clean', 'dark-ir')
    clips, _, labels = official_split('validation', 466)
    model = DomainVideoMAE()
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    model.cuda().to(memory_format=torch.channels_last_3d).eval()
    heads = {'initial': model.head[1]}
    for name in RECIPES:
        state = torch.load(output/name/'best_ema.pt', map_location='cpu', weights_only=True)
        linear = torch.nn.Linear(model.encoder.num_features, 40)
        linear.load_state_dict({key: state[f'head.1.{key}'] for key in ('weight', 'bias')})
        heads[name] = linear.cuda().eval()

    logits = {name: np.empty((len(cases), len(clips), 40), dtype=np.float32) for name in heads}
    for x, _, index in DataLoader(clips, batch_size=8, pin_memory=True):
        x = x.cuda(non_blocking=True)
        for i, case in enumerate(cases):
            view = stress_clip(x, model.mean, case).contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.float16):
                embedding = model.encoder(model.normalized_clip(view))
                for name, head in heads.items():
                    logits[name][i, index.numpy()] = head(embedding).float().cpu().numpy()

    report = {}
    for name, scores in logits.items():
        row = {case: metrics(labels, scores[i]) for i, case in enumerate(cases)}
        for score in row.values():
            score.pop('confusion')
        delta = scores[1] - scores[0]
        row['dark_ir_logit_shift_rms'] = float(np.sqrt(np.mean(np.square(delta - delta.mean(1, keepdims=True)))))
        report[name] = row
    write(output/'ir-probe.json', {'samples': len(clips), 'ir_gain': .5, 'results': report})
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Training artifacts directory')
    parser.add_argument('--packages', type=Path, help='Optional full raw validation and package directory')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--ir-probe', action='store_true', help='Evaluate 50% IR exposure on validation cache')
    args = parser.parse_args()
    if args.extract_only and (args.packages or args.ir_probe):
        parser.error('Evaluation requires fitted heads')
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha(WEIGHTS)
    if not (output / 'feature-source.json').is_file():
        extract(output, source_hash)
    if not args.extract_only:
        make_candidates(output, source_hash)
    if args.packages:
        from eval_phase_order_v8 import RAW, compare, read

        compare(output, args.packages.resolve(), read(RAW/'manifest.json'),
                arms=dict.fromkeys(RECIPES, 'videomae2_dual_v8'), columns=COLUMNS,
                title='冻结骨干与红外亮度一致性', baseline=BASE, training_sources=(Path(__file__).name,))
    if args.ir_probe and not (output/'ir-probe.json').is_file():
        probe_ir(output)


if __name__ == '__main__':
    main()
