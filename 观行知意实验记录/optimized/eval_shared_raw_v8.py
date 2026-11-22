"""Evaluate compatible V8 packages from one decode of each original validation clip."""
import csv
from collections import Counter
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from eval_local_motion_v8 import predictions, read
from package_domain_v8 import ROOT, package
from train_videomae import sha, write


def stage_package(runs, out, name, architecture):
    folder = out / name
    if not folder.is_dir():
        with TemporaryDirectory(dir=out) as temporary:
            staged = Path(temporary) / name
            package(runs, staged, {'mode': name, 'architecture': architecture, 'fusion': False})
            staged.rename(folder)
    return folder


@torch.inference_mode()
def shared_raw(runs, out, dataset, manifest, arms, baseline):
    out.mkdir(parents=True, exist_ok=True)
    pending = {name: stage_package(runs, out, name, arch) for name, arch in arms.items()
               if not (out/name/'metrics.json').is_file()}
    if not pending:
        return

    reference_dir = next(iter(pending.values()))/'code'
    sys.path.insert(0, str(reference_dir))
    from inference import Model

    samples = manifest['samples']
    sample_ids = [sample.get('sample_id', sample['file']+'.npy') for sample in samples]
    labels = dict(zip(sample_ids, (int(sample['label']) for sample in samples)))
    starting = predictions(baseline/'validation.csv')
    if len(sample_ids) != len(labels) or len(labels) != 466 or starting.keys() != labels.keys():
        raise ValueError('Expected matching 466-sample official validation manifests')
    errors = Counter((labels[key], starting[key]) for key in labels if labels[key] != starting[key])
    pairs = [pair for pair, _ in errors.most_common(5)]

    common_config = None
    models, probes = {}, {}
    for name, folder in pending.items():
        code = folder/'code'
        for file in ('inference.py','depth_ir_io.py','build_cache.py','crop_policy.py','videomae_v8.py'):
            reference = reference_dir/file
            if sha(code/file) != sha(reference):
                raise ValueError(f'Shared preprocessing source differs for {name}')
        config = read(code/'config.json')
        detector_hash = sha(code/config['detector_file']) if config.get('person_crop') else None
        signature = ({key: value for key, value in config.items() if key != 'architecture'}, detector_hash)
        if common_config is None:
            common_config = signature
        if signature != common_config or config.get('architecture') != arms[name]:
            raise ValueError(f'Incompatible input configuration in {name}')
        check = folder/'offline-check.json'
        subprocess.run([sys.executable, str(ROOT/'tools/offline_probe.py'), '--package', str(folder/'code.zip'),
                        '--weights', str(folder/'model.pt'), '--sample', str(dataset/samples[0]['file']),
                        '--output', str(check)], check=True)
        probes[name] = read(check)
        config.update(weights_path=str(folder/'model.pt'), package_dir=str(code))
        models[name] = Model(config, 'cuda')
        if models[name].components:
            raise ValueError('Shared raw evaluation requires single classifiers')
        if not probes[name]['passed'] or probes[name]['parameters'] != models[name].parameters:
            raise ValueError(f'Offline probe failed for {name}')

    scores = {name: [] for name in models}
    shared_preprocessor = next(iter(models.values()))
    for index, sample in enumerate(samples):
        path = (dataset / sample['file']).resolve()
        if not path.is_relative_to(dataset.resolve()):
            raise ValueError('Sample path escapes dataset')
        pixels = shared_preprocessor.prepare(path)
        for name, model in models.items():
            prediction = model._scores(model.model, pixels, bool(model.config.get('flip_tta', False)))
            values = prediction.cpu().numpy()
            if index < 3:
                np.testing.assert_array_equal(values, np.asarray(model.predict(path), dtype=values.dtype))
            if values.shape != (40,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid scores from {name}')
            scores[name].append(int(values.argmax()))
        if (index+1) % 50 == 0 or index+1 == len(samples):
            print(f'Shared original-image predictions {index+1}/{len(samples)} for {len(models)} candidates', flush=True)

    for name, folder in pending.items():
        csv_file = folder/'validation.csv'
        with csv_file.open('w', encoding='utf-8', newline='') as target:
            writer = csv.writer(target)
            writer.writerow(['sample_id','predicted_class'])
            writer.writerows(zip(sample_ids, scores[name]))
        code = folder/'code'
        subprocess.run([sys.executable, str(code/'analyze_predictions.py'), '--predictions', str(csv_file),
                        '--manifest', str(dataset/'manifest.json'), '--output', str(folder/'report')], check=True)
        accuracy = read(folder/'report/report.json')['metrics']
        current = dict(zip(sample_ids, scores[name]))
        mistakes = Counter((labels[key], current[key]) for key in labels if labels[key] != current[key])
        result = dict(accuracy, mode=name, parameters=probes[name]['parameters'],
                      corrected=sum(current[k] == labels[k] != starting[k] for k in labels),
                      regressed=sum(starting[k] == labels[k] != current[k] for k in labels),
                      error_counts=[mistakes[pair] for pair in pairs], weights_sha256=sha(folder/'model.pt'),
                      code_sha256=sha(folder/'code.zip'), linux_verified=False)
        write(folder/'metrics.json', result)
    write(runs/'shared-raw-check.json', {'samples': len(samples), 'candidates': list(models),
                                        'single_api_exact_comparisons_per_candidate': 3})
