"""Compare clean validation and matched synthetic sensor perturbations."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_phase_order_v8 import RAW, compare, read
from package_domain_v8 import ROOT
from sensor_corruption_v8 import stress_clip
from train_videomae import Clips, metrics, write
from videomae_v8 import DomainVideoMAE


INIT = ROOT/'runs/v8-teacher-transfer-20261122/kd-non-target/best_ema.pt'
BASE = ROOT/'submission/v8-teacher-transfer-20261122/kd-non-target'
ARMS = dict.fromkeys(('control', 'augment', 'consistency', 'adapters', 'depth-adapters'), 'videomae2_dual_v8')
COLUMNS = ('普通续训', '传感器增强', '增强与一致性蒸馏', '仅适配器加增强', '深度专项适配器')
CASES = ('clean', 'dark-ir', 'depth-gap')


@torch.inference_mode()
def synthetic_probe(runs):
    models = {}
    for name, path in [('initial', INIT), *((arm, runs/arm/'best_ema.pt') for arm in ARMS)]:
        model = DomainVideoMAE()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        models[name] = model.cuda().to(memory_format=torch.channels_last_3d).eval()
    data = Clips(ROOT/'cache/crop16-192-v3', 'validation', frames=16)
    if len(data) != 466 or len({row['sample_id'] for row in data.rows}) != 466:
        raise ValueError('Expected complete official public validation cache')
    labels = np.empty(len(data), dtype=np.int64)
    probabilities = {name: {case: np.empty((len(data), 40), dtype=np.float32) for case in CASES} for name in models}
    for clips, y, indices in DataLoader(data, batch_size=8, pin_memory=True):
        x = clips.cuda(non_blocking=True)
        labels[indices.numpy()] = y.numpy()
        for case in CASES:
            changed = stress_clip(x, models['initial'].mean, case).contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.float16):
                for name, model in models.items():
                    p = model(changed).float().softmax(1).cpu().numpy()
                    probabilities[name][case][indices.numpy()] = p
    rows = {}
    for name, cases in probabilities.items():
        rows[name] = {}
        for case, p in cases.items():
            score = metrics(labels, p)
            score.pop('confusion')
            rows[name][case] = score
        print(name, rows[name], flush=True)
    return {'samples': len(data), 'ir_gain': .5, 'depth_gap_side_fraction': .4, 'results': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--probe-only', action='store_true')
    operation.add_argument('--only', choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.probe_only:
        write(args.runs/'synthetic-stress.json', synthetic_probe(args.runs))
        return
    arms = {args.only: ARMS[args.only]} if args.only else ARMS
    compare(args.runs, args.output, read(RAW/'manifest.json'), arms=arms, columns=COLUMNS,
            title='深度与红外的传感器扰动适配', baseline=BASE, report=args.only is None)


if __name__ == '__main__':
    main()
