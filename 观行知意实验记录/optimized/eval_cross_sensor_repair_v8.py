"""Evaluate IR-conditioned local depth feature repair."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eval_local_reliability_v8 import sensor_probe
from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE
from eval_shared_raw_v8 import shared_raw
from package_domain_v8 import ROOT
from sensor_corruption_v8 import local_sensor_coverage, stress_clip
from train_videomae import Clips, write
from videomae_v8 import DomainVideoMAE, LocalModalityBalance


ARMS = dict.fromkeys(('repair-ce', 'repair-latent', 'repair-strong'), 'videomae2_localrepair_v8')
COLUMNS = ('动作标签', '特征修复权重 0.25', '特征修复权重 0.75')
DEFECTS = ('depth-gap', 'short-offset-gap')


@torch.inference_mode()
def feature_probe(runs, manifest):
    dataset = Clips(ROOT/'cache/crop16-192-v3', 'validation', frames=16)
    truth = {r.get('sample_id', r['file']+'.npy'): int(r['label']) for r in manifest['samples']}
    observed = {row['sample_id']: int(row['label']) for row in dataset.rows}
    if len(dataset) != 466 or len(truth) != 466 or observed != truth:
        raise ValueError('Validation cache differs from the release split')
    loader = DataLoader(dataset, batch_size=8, pin_memory=True)
    result = {}
    for name in ARMS:
        model = DomainVideoMAE(variant='localrepair').cuda().eval()
        state = torch.load(runs/name/'best_ema.pt', weights_only=True, map_location='cpu')
        model.load_state_dict(state)
        patch = model.encoder.patch_embed
        balance = patch.modality_balance
        totals = torch.zeros((len(DEFECTS), 4), device='cuda', dtype=torch.float64)
        for x, _, _ in loader:
            clean = x.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                clean_depth, clean_ir = patch.streams(model.normalized_clip(clean))
                clean_fused, _ = LocalModalityBalance.fuse(balance, clean_depth, clean_ir)
                for i, case in enumerate(DEFECTS):
                    view = stress_clip(clean, model.mean, case)
                    depth, ir = patch.streams(model.normalized_clip(view))
                    before, _ = LocalModalityBalance.fuse(balance, depth, ir)
                    after, _, residual = balance.repair_parts(depth, ir)
                    coverage = local_sensor_coverage(clean, view)[...,0]
                    target = clean_fused.float()
                    depth_residual = clean_depth.float() - depth.float()
                    totals[i] += torch.stack((
                        coverage.sum(),
                        ((before.float()-target).square().mean(-1)*coverage).sum(),
                        ((after.float()-target).square().mean(-1)*coverage).sum(),
                        ((residual.float()-depth_residual).square().mean(-1)*coverage).sum(),
                    ))
        row = {}
        for i, case in enumerate(DEFECTS):
            count, before, after, residual = totals[i].tolist()
            row[case] = {'fused_mse_before': before/count, 'fused_mse_after': after/count,
                         'depth_residual_mse': residual/count, 'relative_fused_mse_change': (after-before)/before}
        result[name] = row
        print(name, row, flush=True)
    write(runs/'feature-probe.json', {'samples': len(dataset), 'cases': list(DEFECTS), 'results': result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, out = args.runs.resolve(), args.output.resolve()
    manifest = read(RAW/'manifest.json')
    shared_raw(runs, out, RAW, manifest, ARMS, BASE)
    compare(runs, out, manifest, arms=ARMS, columns=COLUMNS, title='利用 IR 修复局部深度特征', baseline=BASE)
    sensor_probe(runs, manifest, variants=dict.fromkeys(ARMS, 'localrepair'))
    feature_probe(runs, manifest)


if __name__ == '__main__':
    main()
