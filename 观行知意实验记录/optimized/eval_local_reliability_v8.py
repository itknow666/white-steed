"""Compare local depth/IR reliability and diagnose responses to sensor defects."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE, INIT
from eval_shared_raw_v8 import shared_raw
from package_domain_v8 import ROOT
from sensor_corruption_v8 import local_sensor_coverage, stress_clip
from train_videomae import Clips, metrics, save, sha, write
from videomae_v8 import DomainVideoMAE


VARIANTS = {'adapters-defect': 'plain', 'local-task': 'local', 'local-quality': 'local',
            'local-quality-last': 'local', 'local-quality-conservative': 'local'}
ARMS = {name: 'videomae2_local_v8' if variant == 'local' else 'videomae2_dual_v8'
        for name, variant in VARIANTS.items()}
COLUMNS = ('原适配器加缺损', '局部配比只用动作标签', '局部配比加缺损位置', '缺损位置组末轮', '末轮加保守阈值')
CASES = ('clean', 'dark-ir', 'depth-gap', 'short-offset-gap')


def materialize_last_variants(runs):
    source = runs/'local-quality'
    last = source/'last_ema.pt'
    state = torch.load(last, weights_only=True, map_location='cpu')
    history = (source/'history.jsonl').read_text(encoding='utf-8').splitlines()
    row = json.loads(history[-1])
    source_hash = sha(last)
    for name, shift in (('local-quality-last', 0.), ('local-quality-conservative', -1.5)):
        folder = runs/name
        if (folder/'summary.json').is_file():
            continue
        folder.mkdir(exist_ok=True)
        candidate = dict(state)
        field = 'encoder.patch_embed.modality_balance.quality.3.bias'
        candidate[field] = state[field] + shift
        save(folder/'best_ema.pt', candidate)
        write(folder/'config.json', {'source': 'local-quality/last_ema.pt', 'source_sha256': source_hash,
                                     'epoch': row['epoch'], 'quality_logit_bias_shift': shift})
        write(folder/'summary.json', {'mode': name, 'source_epoch': row['epoch']})


@torch.inference_mode()
def sensor_probe(runs, manifest, *, variants=VARIANTS):
    dataset = Clips(ROOT/'cache/crop16-192-v3', 'validation', frames=16)
    truth = {r.get('sample_id', r['file']+'.npy'): int(r['label']) for r in manifest['samples']}
    observed = {r['sample_id']: int(r['label']) for r in dataset.rows}
    if len(dataset) != 466 or len(truth) != 466 or truth != observed:
        raise ValueError('Validation cache differs from the release split')
    labels = np.asarray([row['label'] for row in dataset.rows], dtype=np.int64)
    loader = DataLoader(dataset, batch_size=8, pin_memory=True)
    results = {}
    for name, variant in {'initial': 'plain', **variants}.items():
        state = INIT if name == 'initial' else runs/name/'best_ema.pt'
        model = DomainVideoMAE(variant=variant)
        model.load_state_dict(torch.load(state, weights_only=True, map_location='cpu'), strict=True)
        model.cuda().to(memory_format=torch.channels_last_3d).eval()
        local = variant in ('local', 'localcontrast')
        scores = np.empty((len(CASES), len(dataset), 40), dtype=np.float32)
        # Per case: inside coverage, outside coverage, probabilities, and signed gain.
        totals = torch.zeros((len(CASES), 5), dtype=torch.float64, device='cuda')
        for clip, _, index in loader:
            clip = clip.cuda(non_blocking=True)
            for i, case in enumerate(CASES):
                view = stress_clip(clip, model.mean, case).contiguous(memory_format=torch.channels_last_3d)
                with torch.autocast('cuda', dtype=torch.float16):
                    if local:
                        logits, quality = model.forward_with_local_quality(view)
                    else:
                        logits = model(view)
                scores[i, index.numpy()] = logits.float().cpu().numpy()
                if local:
                    probabilities = quality.sigmoid()
                    sensor = int(case == 'dark-ir')
                    region = local_sensor_coverage(clip, view)[...,sensor] >= .5
                    bad = probabilities[...,sensor]
                    gain = model.encoder.patch_embed.modality_balance.corruption_gain(quality)[...,0]
                    if sensor == 0:
                        gain = -gain
                    totals[i] += torch.stack((region.sum(), (~region).sum(),
                                              (bad*region).sum(), (bad*(~region)).sum(),
                                              (gain*region).sum()))
        row = {}
        for i, case in enumerate(CASES):
            row[case] = metrics(labels, scores[i])
            row[case].pop('confusion')
            changed = scores[0].argmax(1) != scores[i].argmax(1)
            row[case]['changed_predictions'] = int(changed.sum())
            if local:
                inside, outside, inside_bad, outside_bad, inside_gain = totals[i].tolist()
                row[case].update(fault_score_inside=inside_bad/inside if inside else None,
                                 fault_score_outside=outside_bad/outside if outside else None,
                                 clean_sensor_gain_inside=inside_gain/inside if inside else None)
        results[name] = row
        compact = {case: round(r['accuracy'], 3) for case, r in row.items()}
        print(name, compact, flush=True)
    write(runs/'sensor-probe.json', {'samples': len(dataset), 'cases': list(CASES), 'results': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, output = args.runs.resolve(), args.output.resolve()
    manifest = read(RAW/'manifest.json')
    materialize_last_variants(runs)
    shared_raw(runs, output, RAW, manifest, ARMS, BASE)
    compare(runs, output, manifest, arms=ARMS, columns=COLUMNS, title='时空局部双传感器可靠度', baseline=BASE)
    if args.probe:
        sensor_probe(runs, manifest)


if __name__ == '__main__':
    main()
