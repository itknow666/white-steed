"""Fit specialist score weights on official training logits only."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fit_frozen_head_v8 import official_split
from train_videomae import metrics, sha, write


def log_probabilities(logits):
    return torch.from_numpy(logits).softmax(2).clamp_min(1e-12).log().double()


def fit(log_scores, labels, per_class):
    # Give the pretrained dual branch a conservative prior.
    prior = torch.tensor([.84,.10,.06], dtype=torch.float64)[:,None]
    width = 40 if per_class else 1
    logits = torch.nn.Parameter(prior.log().expand(-1,width).clone())
    optimizer = torch.optim.LBFGS([logits], max_iter=80, line_search_fn='strong_wolfe')

    def objective():
        weights = logits.softmax(0)
        predictions = (log_scores * weights[:,None,:]).sum(0)
        reg = .5 * (weights-prior).square().mean()
        return F.cross_entropy(predictions, labels) + reg

    before = float(objective().detach())

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    after = float(objective().detach())
    if not np.isfinite([before,after]).all() or after > before:
        raise ValueError('Invalid fitted mixture')
    return logits.softmax(0).detach().expand(-1,40).contiguous(), before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runs = args.runs.resolve()
    source = runs/'specialist-train-logits.npz'
    _, ids, labels = official_split('train', 1995)
    _, validation_ids, validation_labels = official_split('validation', 466)
    with np.load(source, allow_pickle=False) as train:
        if train['sample_ids'].tolist()!=ids or not np.array_equal(train['labels'],labels):
            raise ValueError('Logit calibration expects only official train rows')
        train_scores = log_probabilities(train['logits'])
    rows = {}
    for name, by_class in (('fitted-global',False), ('fitted-per-class',True)):
        weights,before,after = fit(train_scores,torch.from_numpy(labels),by_class)
        rows[name] = {'log_mix_weights':weights.tolist(), 'train_objective_before':before,
                      'train_objective_after':after}
    # Load validation only after both train-only fits complete.
    with np.load(runs/'specialist-validation.npz', allow_pickle=False) as val:
        if val['sample_ids'].tolist()!=validation_ids or not np.array_equal(val['labels'],validation_labels):
            raise ValueError('Calibration validation logit IDs differ')
        validation_scores = log_probabilities(val['trained_logits'])
    for name, row in rows.items():
        weights = torch.tensor(row['log_mix_weights'], dtype=torch.float64)
        predictions = (validation_scores*weights[:,None,:]).sum(0).numpy()
        score = metrics(validation_labels,predictions)
        score.pop('confusion')
        row.update(score)
        print(name, score, 'mean weights', weights.mean(1).tolist(), flush=True)
    write(runs/'fitted-mixtures.json', {'train_samples':len(labels), 'validation_samples':len(validation_labels),
                                       'train_logits_sha256':sha(source), 'results':rows})


if __name__ == '__main__':
    main()
