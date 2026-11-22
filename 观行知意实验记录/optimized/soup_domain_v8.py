"""Average compatible VideoMAE weights and evaluate the public validation cache."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from run_v8_paper_sprint import ROOT, RUNS, BASE
from train_videomae import Clips, evaluate, sha, write
from videomae_model import VideoMAEClassifier
from videomae_v8 import DomainVideoMAE


def average_states(states):
    keys = states[0].keys()
    if any(state.keys() != keys for state in states):
        raise ValueError('Checkpoint layouts differ')
    merged = {}
    for key in keys:
        tensors = [state[key] for state in states]
        first = tensors[0]
        if any(t.shape != first.shape or t.dtype != first.dtype for t in tensors):
            raise ValueError(f'Tensor layout differs for {key}')
        if key in ('mean', 'std') or not first.is_floating_point():
            if any(not torch.equal(first, t) for t in tensors[1:]):
                raise ValueError(f'Incompatible fixed buffer {key}')
            merged[key] = first
        else:
            merged[key] = torch.stack([t.float() for t in tensors]).mean(0).to(first.dtype)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v7-reference', action='store_true', help='Average the three inputs before dual adaptation')
    args = parser.parse_args()
    transfer43 = RUNS/'transfer43/best_ema.pt'
    transfer44 = RUNS/'transfer44/best_ema.pt'
    source43 = ROOT/'runs/videomae2_s-v7-seed43-20261114/best_ema.pt'
    recipes = {'soup43-v8': (transfer43, BASE),
               'soup43-44': (transfer43, transfer44),
               'soup-three': (transfer43, transfer44, BASE),
               'anchor43-half': (source43, transfer43)}
    if args.v7_reference:
        recipes = {'v7-source-soup': (ROOT/'runs/videomae2_s-v7-20261114/best_ema.pt', source43,
                                     ROOT/'runs/videomae2_s-v7-seed44-20261114/best_ema.pt')}
    torch.set_num_threads(4)
    dataset = Clips(ROOT/'cache/crop16-192-v3', 'validation', frames=16)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True)
    ids = np.asarray([r['sample_id'] for r in dataset.rows])
    if len(set(ids)) != 466:
        raise ValueError('Expected official public validation set')
    for name, recipe in recipes.items():
        folder = RUNS/name
        folder.mkdir()
        states = [torch.load(path, map_location='cpu', weights_only=True) for path in recipe]
        if name == 'anchor43-half':
            # Match the adapter initialization used when migrating seed 43.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                states[0] = DomainVideoMAE(warmstart=states[0]).state_dict()
        merged = average_states(states)
        model = VideoMAEClassifier() if args.v7_reference else DomainVideoMAE()
        model.load_state_dict(merged, strict=True)
        result, probabilities, labels = evaluate(model.cuda(), loader)
        torch.save(merged, folder/'best_ema.pt')
        np.savez_compressed(folder/'best_predictions.npz', probabilities=probabilities, labels=labels, sample_ids=ids)
        write(folder/'config.json', {'domain_variant':'plain','method':'equal checkpoint averaging',
                                     'inputs':[str(p.relative_to(ROOT)) for p in recipe],
                                     'source_sha256':[sha(p) for p in recipe],
                                     'convert_first_to_dual': name == 'anchor43-half'})
        result.pop('confusion')
        write(folder/'summary.json', dict(result, best_epoch=None))
        print(name, result, flush=True)


if __name__ == '__main__':
    main()
