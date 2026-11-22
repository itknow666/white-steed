"""Measure depth/IR specialists and prepare two train-only distillation targets."""
import argparse
from pathlib import Path

import numpy as np
import torch

from cache_teacher_v8 import training_logits
from eval_phase_order_v8 import RAW, read
from eval_sensor_robustness_v8 import INIT
from fit_frozen_head_v8 import official_split
from train_videomae import metrics, sha, write
from videomae_v8 import DomainVideoMAE


NAMES = ('dual', 'depth-expert', 'ir-expert')
VARIANTS = ('plain', 'depthonly', 'ironly')
MIXTURES = {'expert-pair': {'depth-expert':.6, 'ir-expert':.4},
            'dual-with-experts': {'initial':.7, 'depth-expert':.18, 'ir-expert':.12},
            'dual-with-ir': {'initial':.85, 'ir-expert':.15},
            'dual-with-depth': {'initial':.85, 'depth-expert':.15}}


def save_npz(path, **fields):
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **fields)
    temporary.replace(path)


def outputs(dataset, weights):
    logits = np.stack([training_logits(DomainVideoMAE(variant=variant), dataset, path)
                       for variant, path in zip(VARIANTS, weights)])
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite specialist logits')
    return logits


def distributions(logits, labels):
    z = torch.from_numpy(logits) / 2
    ids = torch.from_numpy(labels)[None, :, None].expand(len(z), -1, -1)
    return z.scatter(2, ids, -torch.inf).softmax(2).numpy()


def agreement(logits, labels):
    correct = logits.argmax(2) == labels[None, :]
    return {
        'correct_by_model': dict(zip(NAMES, correct.sum(1).tolist())),
        'only_depth_expert': int((correct[1] & ~correct[2]).sum()),
        'only_ir_expert': int((correct[2] & ~correct[1]).sum()),
        'either_expert': int((correct[1] | correct[2]).sum()),
        'expert_correct_where_dual_wrong': int(((correct[1] | correct[2]) & ~correct[0]).sum()),
    }


def complement_by_class(logits, labels, names):
    correct = logits.argmax(2) == labels[None,:]
    rows = {}
    for index,name in enumerate(names):
        match = labels == index
        rows[name] = {
            'samples': int(match.sum()),
            'only_depth_expert': int((match & correct[1] & ~correct[2]).sum()),
            'only_ir_expert': int((match & correct[2] & ~correct[1]).sum()),
            'either_expert_correct_where_dual_wrong': int((match & (correct[1] | correct[2]) & ~correct[0]).sum()),
        }
    return rows


def make_targets(runs, weights):
    dataset, ids, labels = official_split('train', 1995)
    logits = outputs(dataset, weights)
    save_npz(runs/'specialist-train-logits.npz', sample_ids=ids, labels=labels, logits=logits)
    probabilities = distributions(logits, labels)
    judged = logits[1:].argmax(2) == labels[None,:]
    expert_mean = probabilities[1:].mean(0)
    count = judged.sum(0)[:,None]
    good_experts = (probabilities[1:]*judged[...,None]).sum(0) / count.clip(1)
    all_teachers = .6 * probabilities[0] + .4 * expert_mean
    selected_teachers = np.where(count > 0, .6 * probabilities[0] + .4 * good_experts, probabilities[0])
    for name, target in (('all', all_teachers), ('selected', selected_teachers)):
        if not (np.isfinite(target).all() and np.allclose(target.sum(1), 1)):
            raise ValueError(f'Invalid train-only teacher probabilities for {name}')
        save_npz(runs/f'teacher-{name}.npz', sample_ids=ids, labels=labels,
                 non_target_targets=target.astype(np.float32))
    write(runs/'teacher-sources.json', {'split': 'train', 'samples': len(labels),
                                      'modalities': list(NAMES), 'sha256': [sha(w) for w in weights],
                                      'temperature': 2., 'expert_fraction': .4,
                                      'teacher_all_sha256': sha(runs/'teacher-all.npz'),
                                      'teacher_selected_sha256': sha(runs/'teacher-selected.npz'),
                                      'agreement': agreement(logits, labels)})


def validate_cache(runs, weights):
    dataset, ids, labels = official_split('validation', 466)
    trained = outputs(dataset, weights)
    untouched = np.stack([trained[0], *(training_logits(DomainVideoMAE(variant=kind), dataset, INIT)
                                        for kind in VARIANTS[1:])])
    arrays = torch.from_numpy(trained).softmax(2).numpy()
    baseline_depth, baseline_ir = untouched[1:]
    predictions = {
        'initial': trained[0], 'untrained-depth': baseline_depth, 'untrained-ir': baseline_ir,
        'depth-expert': trained[1], 'ir-expert': trained[2],
    }
    by_model = dict(zip(('initial', *NAMES[1:]), arrays))
    predictions.update({
        name: sum(weight * by_model[arm] for arm, weight in mixture.items())
        for name, mixture in MIXTURES.items()
    })
    results = {}
    for name, p in predictions.items():
        row = metrics(labels,p)
        row.pop('confusion')
        results[name] = row
    write(runs/'specialist-validation.json', {'samples': len(labels), 'agreement': agreement(trained,labels),
                                             'results': results})
    save_npz(runs/'specialist-validation.npz', sample_ids=ids, labels=labels,
             trained_logits=trained, untouched_logits=untouched)
    manifest = read(RAW/'manifest.json')
    write(runs/'expert-class-complement.json', complement_by_class(trained, labels, manifest['classes']))
    print(results, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--split', choices=('train','validation'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs = args.runs.resolve()
    weights = [INIT, *(runs/name/'best_ema.pt' for name in NAMES[1:])]
    if args.split == 'train':
        make_targets(runs, weights)
    else:
        validate_cache(runs, weights)


if __name__ == '__main__':
    main()
