"""Package and verify depth/IR specialists, mixtures and distilled students."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from eval_local_motion_v8 import predictions
from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE, INIT
from eval_shared_raw_v8 import shared_raw
from package_domain_v8 import package, validate
from probe_modality_experts_v8 import MIXTURES
from train_videomae import save, sha, write


MERGES = {'arithmetic-10':.10, 'arithmetic-25':.25}
FITTED = ('fitted-global', 'fitted-per-class')
SINGLES = {'depth-expert':'videomae2_depthonly_v8', 'ir-expert':'videomae2_ironly_v8',
           'dual-control':'videomae2_dual_v8', 'dual-distill-all':'videomae2_dual_v8',
           'dual-distill-selected':'videomae2_dual_v8', **dict.fromkeys(MERGES, 'videomae2_dual_v8')}
DISPLAY = {'depth-expert':'深度专家', 'ir-expert':'IR 专家', 'dual-control':'双路续训对照',
           'dual-distill-all':'所有专家蒸馏', 'dual-distill-selected':'正确专家蒸馏',
           'arithmetic-10':'专家增量各加 0.10', 'arithmetic-25':'专家增量各加 0.25',
           'fitted-global':'训练集全局对数融合', 'fitted-per-class':'训练集逐类对数融合',
           'expert-pair':'双专家直接融合', 'dual-with-experts':'原双路加入两位专家',
           'dual-with-ir':'原双路加入 IR 专家', 'dual-with-depth':'原双路加入深度专家'}


def merge_expert_states(runs):
    pending = {name:scale for name,scale in MERGES.items() if not (runs/name/'summary.json').is_file()}
    if not pending:
        return
    sources = [INIT, *(runs/name/'best_ema.pt' for name in ('depth-expert','ir-expert'))]
    baseline, depth, ir = [torch.load(path, weights_only=True, map_location='cpu') for path in sources]
    if not baseline.keys()==depth.keys()==ir.keys():
        raise ValueError('Specialist weight layouts differ')
    hashes = [sha(path) for path in sources]
    for name,scale in pending.items():
        folder = runs/name
        folder.mkdir(exist_ok=True)
        state = {}
        for key, original in baseline.items():
            if depth[key].shape != original.shape or ir[key].shape != original.shape:
                raise ValueError(f'Specialist tensor shapes differ for {key}')
            if original.is_floating_point():
                state[key] = original + scale*((depth[key]-original)+(ir[key]-original))
            else:
                state[key] = original
        save(folder/'best_ema.pt', state)
        write(folder/'config.json', {'source_sha256':hashes, 'depth_scale':scale, 'ir_scale':scale})
        write(folder/'summary.json', {'mode':name, 'training_steps':0})


def evaluate_mixture(runs, output, name, labels, original, fitted):
    folder = output/name
    if (folder/'metrics.json').is_file():
        return
    specs = [
        {'architecture': SINGLES.get(arm, 'videomae2_dual_v8'), 'weight': weight,
         'weights': INIT if arm == 'initial' else runs/arm/'best_ema.pt'}
        for arm, weight in MIXTURES.get(name, MIXTURES['dual-with-experts']).items()
    ]
    if not folder.exists():
        with TemporaryDirectory(dir=output) as tmp:
            prepared = Path(tmp)/name
            recipe = {'mode': name, 'fusion': True, 'components': specs}
            if name in fitted:
                recipe['log_mix_weights'] = fitted[name]['log_mix_weights']
            package(runs, prepared, recipe)
            prepared.rename(folder)
    raw, probe = validate(folder)
    current = predictions(folder/'validation.csv')
    if current.keys()!=labels.keys():
        raise ValueError(f'Sample IDs differ for mixture {name}')
    mistakes = Counter((labels[k], current[k]) for k in labels if labels[k] != current[k])
    original_errors = Counter((labels[k], original[k]) for k in labels if labels[k] != original[k])
    pairs = [pair for pair, _ in original_errors.most_common(5)]
    checks = {'mode': name, 'parameters': probe['parameters'],
              'corrected': sum(current[k]==labels[k]!=original[k] for k in labels),
              'regressed': sum(original[k]==labels[k]!=current[k] for k in labels),
              'error_counts': [mistakes[pair] for pair in pairs],
              'weights_sha256': sha(folder/'model.pt'), 'code_sha256': sha(folder/'code.zip'),
              'linux_verified': False}
    write(folder/'metrics.json', dict(raw, **checks))


@torch.inference_mode()
def measure_latency(runs, packages, manifest):
    report = read(runs/'raw-comparison.json')
    mixtures = [name for name in report['results'] if name in MIXTURES or name in FITTED]
    best_mixture = max(mixtures, key=lambda name: (report['results'][name]['accuracy'],
                                                   report['results'][name]['macro_f1']))
    folders = {'initial': BASE, best_mixture: packages/best_mixture}
    recommended = report['recommended']
    if recommended not in folders:
        folders[recommended] = packages/recommended
    sys.path.insert(0, str((packages/best_mixture/'code').resolve()))
    from inference import Model

    paths = [RAW/manifest['samples'][index]['file']
             for index in np.linspace(0, len(manifest['samples'])-1, 53, dtype=int)]
    timings = {}
    for name, folder in folders.items():
        config = read(folder/'code/config.json')
        config.update(weights_path=str(folder/'model.pt'), package_dir=str(folder/'code'))
        model = Model(config,'cuda')
        for path in paths[:3]:
            model.predict(path)
        durations = []
        for path in paths[3:]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(path)
            torch.cuda.synchronize()
            durations.append(1000 * (time.perf_counter()-start))
        timings[name] = {'mean_ms':float(np.mean(durations)), 'median_ms':float(np.median(durations)),
                         'p95_ms':float(np.percentile(durations,95))}
        del model
        print(name, timings[name], flush=True)
    write(runs/'local-latency.json', {'samples':50,'warmup':3,'includes_crop_and_decode':True,
                                     'device':torch.cuda.get_device_name(), 'results':timings})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--skip-latency', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, output = args.runs.resolve(), args.output.resolve()
    manifest = read(RAW/'manifest.json')
    labels = {row.get('sample_id', row['file']+'.npy'):int(row['label']) for row in manifest['samples']}
    original = predictions(BASE/'validation.csv')
    if len(labels)!=466 or original.keys()!=labels.keys():
        raise ValueError('Baseline and release manifest differ')
    cache = read(runs/'specialist-validation.json')['results']
    selected = sorted(MIXTURES, key=lambda name:(cache[name]['accuracy'],cache[name]['macro_f1']), reverse=True)[:2]
    fitted = read(runs/'fitted-mixtures.json')['results']
    best_fitted = max(FITTED, key=lambda name:(fitted[name]['accuracy'],fitted[name]['macro_f1']))
    write(runs/'raw-selection.json', {'mixtures_ranked_on_cache':selected,
                                    'fitted_ranked_on_cache':best_fitted,
                                    'single_candidates':list(SINGLES)})
    selected = [*selected,best_fitted]
    merge_expert_states(runs)
    shared_raw(runs, output, RAW, manifest, SINGLES, BASE)
    torch.cuda.empty_cache()
    for name in selected:
        evaluate_mixture(runs, output, name, labels, original, fitted)
    arms = {**SINGLES, **dict.fromkeys(selected, 'ensemble')}
    compare(runs, output, manifest, arms=arms, columns=tuple(DISPLAY[name] for name in arms),
            title='深度与 IR 单模态专家和迁移', baseline=BASE)
    if not args.skip_latency:
        measure_latency(runs, output, manifest)


if __name__ == '__main__':
    main()
