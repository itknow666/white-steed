"""Compare matched single-model performance when cached IR is synthetically darker."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from train_videomae import Clips, evaluate, write
from videomae_v8 import DomainVideoMAE


class DarkIR(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, label, row = self.data[index]
        x[3].mul_(.55)
        return x, label, row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    dataset = DarkIR(Clips(a.cache, 'validation', frames=16))
    if len(dataset) != 466:
        raise ValueError('Expected 466 validation clips')
    loader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True)
    clean = json.loads((a.runs/'comparison.json').read_text(encoding='utf-8'))['scores']
    results = {}
    for group in ('v8-original', 'gate', 'masked', 'distill'):
        if group not in clean:
            continue
        path = a.baseline if group == 'v8-original' else a.runs/group/'best_ema.pt'
        model = DomainVideoMAE(variant='plain' if group == 'v8-original' else 'gate')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        result, _, _ = evaluate(model.cuda(), loader)
        result.pop('confusion')
        previous = clean[group]['single']
        results[group] = {'clean': previous, 'dim_ir': result,
                          'accuracy_drop_pp': previous['accuracy']-result['accuracy']}
        print(group, results[group], flush=True)
    write(a.runs/'ir-stress.json', {'results': results, 'multiplier': .55,
                                   'samples': 466, 'evaluation': 'cached single classifier'})


if __name__ == '__main__':
    main()
