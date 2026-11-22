"""Evaluate non-target distillation ablations and averaged checkpoints."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_phase_order_v8 import RAW, compare, read
from package_domain_v8 import ROOT
from soup_domain_v8 import average_states
from train_videomae import Clips, evaluate, save, sha, write
from videomae_v8 import DomainVideoMAE


TRANSFER = ROOT/'runs/v8-teacher-transfer-20261122'
BASE = ROOT/'submission/v8-teacher-transfer-20261122/kd-non-target'
ANCHOR = ROOT/'runs/v8-paper-sprint-20261122/anchor43-half/best_ema.pt'
ARMS = dict.fromkeys(('seed44', 'weight-015', 'weight-060', 'last-four',
                      'average-seed42-43', 'average-seed42-43-44', 'average-anchor-42', 'rescaled-three'),
                   'videomae2_dual_v8')
COLUMNS = ('种子 44', '蒸馏系数 0.15', '蒸馏系数 0.60', '仅末四层', '两种子平均', '三种子平均',
           '起点与种子 42 平均', '三种子平均后恢复更新幅度')


def rescale_update(merged, states, anchor):
    """Match the RMS size of the source updates without using validation labels."""
    squared_mean, squared_sources = 0., 0.
    movable = [key for key, value in merged.items() if value.is_floating_point() and key not in ('mean', 'std')]
    for key in movable:
        start = anchor[key].float()
        squared_mean += float((merged[key].float() - start).square().sum())
        squared_sources += sum(float((state[key].float() - start).square().sum()) for state in states)
    if squared_mean <= 0:
        raise ValueError('Cannot rescale an unchanged average')
    multiplier = (squared_sources / (len(states) * squared_mean)) ** .5
    for key in movable:
        merged[key] = torch.lerp(anchor[key], merged[key], multiplier)
    return multiplier


def create_averages(runs, evaluate_cache):
    seed42 = TRANSFER/'kd-non-target/best_ema.pt'
    seed43 = TRANSFER/'kd-non-target-seed43/best_ema.pt'
    seed44 = runs/'seed44/best_ema.pt'
    recipes = {'average-seed42-43': (seed42, seed43),
               'average-seed42-43-44': (seed42, seed43, seed44),
               'average-anchor-42': (ANCHOR, seed42),
               'rescaled-three': (seed42, seed43, seed44)}
    if evaluate_cache:
        data = Clips(ROOT/'cache/crop16-192-v3', 'validation', frames=16)
        loader = DataLoader(data, batch_size=8, pin_memory=True)
        ids = np.array([row['sample_id'] for row in data.rows])
        if len(ids) != 466 or len(set(ids)) != 466:
            raise ValueError('Expected 466 unique validation samples')
        model = DomainVideoMAE().cuda()
    for name, sources in recipes.items():
        folder = runs/name
        if (folder/'summary.json').is_file() or any(not path.is_file() for path in sources):
            continue
        if seed44 in sources and not (runs/'seed44/summary.json').is_file():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder/'config.json').is_file():
            states = [torch.load(path, map_location='cpu', weights_only=True) for path in sources]
            merged = average_states(states)
            config = {'mode': name, 'method': 'equal checkpoint averaging',
                      'sources': [str(path.relative_to(ROOT)) for path in sources],
                      'source_sha256': [sha(path) for path in sources]}
            if name == 'rescaled-three':
                anchor = torch.load(ANCHOR, map_location='cpu', weights_only=True)
                config.update(method='RMS norm restoration after averaging', anchor_sha256=sha(ANCHOR),
                              update_multiplier=rescale_update(merged, states, anchor))
            save(folder/'best_ema.pt', merged)
            write(folder/'config.json', config)
        if evaluate_cache:
            state = torch.load(folder/'best_ema.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            result, probabilities, labels = evaluate(model, loader)
            result.pop('confusion')
            np.savez_compressed(folder/'best_predictions.npz', probabilities=probabilities, labels=labels, sample_ids=ids)
            write(folder/'summary.json', dict(result, best_epoch=None))
            print(name, result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Assemble available checkpoints on CPU')
    parser.add_argument('--only', choices=ARMS, help='Verify one completed candidate independently')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.only:
        create_averages(args.runs.resolve(), evaluate_cache=not args.prepare_only)
    if args.prepare_only:
        return
    arms = {args.only:ARMS[args.only]} if args.only else ARMS
    compare(args.runs, args.output, read(RAW/'manifest.json'), arms=arms, columns=COLUMNS,
            title='非目标蒸馏复跑与权重平均', baseline=BASE, training_sources=('cache_teacher_v8.py',), report=args.only is None)


if __name__ == '__main__':
    main()
