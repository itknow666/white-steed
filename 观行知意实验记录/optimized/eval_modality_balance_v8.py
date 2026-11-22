"""Validate depth/IR feature-channel balance on the official holdout."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE, INIT
from package_domain_v8 import ROOT
from sensor_corruption_v8 import stress_clip
from train_videomae import Clips, metrics, write
from videomae_v8 import DomainVideoMAE


VARIANTS = {'adapters-control': 'dual', 'static-balance': 'balance',
            'clip-balance': 'film', 'clip-balance-ir': 'film', 'clip-balance-defect': 'film'}
ARMS = {name: f'videomae2_{variant}_v8' for name, variant in VARIANTS.items()}
COLUMNS = ('原适配器对照', '固定通道比例', '按片段估计比例', '片段条件与 IR 亮度增强', '片段条件与传感器缺损增强')
CASES = ('clean', 'dark-ir', 'depth-gap')


@torch.inference_mode()
def probe(runs, arms, *, strength_probe=False):
    dataset = Clips(ROOT/'cache/crop16-192-v3', 'validation', frames=16)
    official = read(RAW/'manifest.json')['samples']
    truth = {r.get('sample_id', r['file']+'.npy'): int(r['label']) for r in official}
    observed = {r['sample_id']: int(r['label']) for r in dataset.rows}
    if len(dataset) != 466 or len(truth) != 466 or observed != truth:
        raise ValueError('Validation cache differs from the release split')
    labels = np.asarray([row['label'] for row in dataset.rows], dtype=np.int64)
    loader = DataLoader(dataset, batch_size=8, pin_memory=True)
    sources = {'initial': ('dual', INIT, 1)}
    sources.update({name: (VARIANTS[name], runs/name/'best_ema.pt', 1) for name in arms})
    if strength_probe:
        checkpoint = runs/'clip-balance-defect'/'best_ema.pt'
        sources = {f'gain-{strength}x': ('film', checkpoint, strength) for strength in (1, 4, 12)}
    results = {}
    for name, (variant, state, strength) in sources.items():
        model = DomainVideoMAE(variant='plain' if variant == 'dual' else variant)
        model.load_state_dict(torch.load(state, weights_only=True, map_location='cpu'), strict=True)
        model = model.cuda().to(memory_format=torch.channels_last_3d).eval()
        balance = getattr(model.encoder.patch_embed, 'modality_balance', None)
        if strength != 1:
            # Scale the pre-tanh preference; the original gain limit of 0.5 still applies.
            balance.prior.mul_(strength)
            balance.condition[-1].weight.mul_(strength)
            balance.condition[-1].bias.mul_(strength)
        scores = np.empty((len(CASES), len(dataset), 40), dtype=np.float32)
        if balance is not None:
            gains = np.empty((len(CASES), len(dataset), model.encoder.num_features), dtype=np.float32)
            recent = []

            def record_gain(module, arguments):
                recent.append(module.gains(*arguments).float().squeeze(-2).cpu().numpy())

            hook = balance.register_forward_pre_hook(record_gain)
        for clips, _, index in loader:
            clips = clips.cuda(non_blocking=True)
            for i, case in enumerate(CASES):
                view = stress_clip(clips, model.mean, case).contiguous(memory_format=torch.channels_last_3d)
                with torch.autocast('cuda', dtype=torch.float16):
                    scores[i, index.numpy()] = model(view).float().cpu().numpy()
                if balance is not None:
                    gains[i, index.numpy()] = recent.pop()
        if balance is not None:
            hook.remove()
        row = {case: metrics(labels, scores[i]) for i, case in enumerate(CASES)}
        for value in row.values():
            value.pop('confusion')
        row['changed_predictions'] = int((scores[0].argmax(1) != scores[1].argmax(1)).sum())
        if balance is not None:
            row.update(gain_abs_mean=float(np.abs(gains[0]).mean()),
                       gain_changed_abs_mean=float(np.abs(gains[1] - gains[0]).mean()),
                       gain_depth_gap_abs_mean=float(np.abs(gains[2] - gains[0]).mean()),
                       gain_across_clip_std=float(gains[0].std(axis=0, dtype=np.float64).mean()))
        results[name] = row
        print(name, row, flush=True)
    conditions = {'samples': len(dataset), 'ir_gain': .5, 'depth_gap_side_fraction': .4, 'results': results}
    if strength_probe:
        conditions['source'] = 'clip-balance-defect'
        write(runs/'strength-probe.json', conditions)
    else:
        write(runs/'ir-probe.json', conditions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--only', choices=ARMS)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--strength-probe', action='store_true', help='Compare amplified gains on the validation cache')
    parser.add_argument('--shared-raw', action='store_true', help='Decode original frames once for compatible packages')
    args = parser.parse_args()
    if not (args.probe or args.strength_probe or args.output):
        parser.error('Specify --output to compare original clips, or --probe for IR exposure')
    torch.set_num_threads(4)
    selected = {args.only: ARMS[args.only]} if args.only else ARMS
    if args.output:
        manifest = read(RAW/'manifest.json')
        if args.shared_raw:
            from eval_shared_raw_v8 import shared_raw

            shared_raw(args.runs.resolve(), args.output.resolve(), RAW, manifest, selected, BASE)
        compare(args.runs.resolve(), args.output.resolve(), manifest,
                arms=selected, columns=COLUMNS, title='深度与 IR 通道适配',
                baseline=BASE, report=args.only is None)
    if args.probe:
        probe(args.runs.resolve(), selected)
    if args.strength_probe:
        probe(args.runs.resolve(), selected, strength_probe=True)


if __name__ == '__main__':
    main()
