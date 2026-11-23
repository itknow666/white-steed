"""Prepare train-only non-target distillation teachers with matched IR budgets."""
import argparse
from pathlib import Path

import numpy as np
import torch

from fit_frozen_head_v8 import official_split
from package_domain_v8 import ROOT
from train_videomae import sha, write


EXPERTS = ROOT/'runs/v8-modality-experts-20261123/specialist-train-logits.npz'
ORIGINAL = ROOT/'runs/v8-teacher-transfer-20261122/teacher-train.npz'
FRACTION = .40
QUOTA = .20


def load(path, ids, labels):
    with np.load(path, allow_pickle=False) as source:
        if source['sample_ids'].tolist() != list(ids) or not np.array_equal(source['labels'], labels):
            raise ValueError(f'{path.name}: teacher IDs and labels must match the official train split')
        return {key: source[key] for key in source.files if key not in ('sample_ids', 'labels')}


def margin(logits, labels):
    competitors = logits.copy()
    rows = np.arange(len(labels))
    competitors[rows, labels] = -np.inf
    return logits[rows, labels] - competitors.max(axis=1)


def percentile(values):
    # Equal margins get the same percentile.
    ordered = np.sort(values)
    return (np.searchsorted(ordered, values, side='left') +
            np.searchsorted(ordered, values, side='right')) / (2 * len(values))


def non_target(logits, labels):
    values = torch.tensor(logits, dtype=torch.float32)
    values[torch.arange(len(labels)), torch.from_numpy(labels)] = -1e4
    return (values / 2).softmax(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    _, ids, labels = official_split('train', 1995)
    labels = np.asarray(labels, dtype=np.int64)
    specialist = load(EXPERTS, ids, labels)['logits']
    original = load(ORIGINAL, ids, labels)
    if (specialist.shape != (3, 1995, 40) or
            any(array.shape != (1995, 40) for array in original.values()) or
            not all(np.isfinite(a).all() for a in [specialist, *original.values()])):
        raise ValueError('Invalid training teacher logits')
    dual, _, ir = specialist
    dual_margin, ir_margin = margin(dual, labels), margin(ir, labels)
    dual_correct, ir_correct = dual.argmax(1) == labels, ir.argmax(1) == labels
    routed = np.zeros(len(labels), dtype=bool)
    balanced_random = routed.copy()
    rng = np.random.default_rng(42)
    per_class = {}
    for category in range(40):
        idx = np.flatnonzero(labels == category)
        eligible = ir_correct[idx]
        count = min(int(np.ceil(QUOTA * len(idx))), int(eligible.sum()))
        gain = percentile(ir_margin[idx]) - percentile(dual_margin[idx])
        rescue = ~dual_correct[idx]
        order = np.lexsort((idx, -gain, -rescue.astype(int)))
        chosen = idx[order[eligible[order]][:count]]
        control = rng.choice(idx[eligible], size=count, replace=False)
        routed[chosen] = True
        balanced_random[control] = True
        per_class[str(category)] = dict(clips=len(idx), ir_correct=int(eligible.sum()), selected=count)
    assert routed.sum() == balanced_random.sum() and np.all(ir_correct[routed | balanced_random])
    baseline = non_target(original['video_logits'],labels).lerp(non_target(original['r34_logits'],labels), .25)
    expert = non_target(ir, labels)
    weights = {'ir-uniform': np.full(len(labels), FRACTION * routed.mean()),
               'ir-class-random': FRACTION * balanced_random, 'ir-routed': FRACTION * routed}
    assert np.allclose([x.sum() for x in weights.values()], weights['ir-routed'].sum())
    cache_info = {}
    for name, weight in weights.items():
        tensor = torch.from_numpy(weight.astype(np.float32))[:, None]
        targets = baseline.lerp(expert, tensor).numpy()
        assert np.isfinite(targets).all() and np.allclose(targets.sum(1), 1)
        assert not targets[np.arange(len(labels)), labels].any()
        path = args.output/f'{name}.npz'
        with path.open('wb') as out:
            np.savez_compressed(out, sample_ids=ids, labels=labels, non_target_targets=targets)
        cache_info[name] = {'sha256': sha(path), 'nonzero': int(np.count_nonzero(weight)),
                            'mean_ir_weight': float(weight.mean())}
    write(args.output/'teacher-routing.json', {
        'split': 'train', 'samples': len(labels), 'class_target_cap_fraction': QUOTA,
        'source_sha256': {'original': sha(ORIGINAL), 'experts': sha(EXPERTS)},
        'routed_correct_when_dual_wrong': int((routed & ~dual_correct).sum()),
        'random_correct_when_dual_wrong': int((balanced_random & ~dual_correct).sum()),
        'available_ir_rescues': int((ir_correct & ~dual_correct).sum()),
        'shared_selected_rows': int((routed & balanced_random).sum()),
        'per_class': per_class, 'caches': cache_info,
    })
    print(cache_info, flush=True)


if __name__ == '__main__':
    main()
