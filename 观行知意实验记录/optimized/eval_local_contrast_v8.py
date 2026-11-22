"""Validate spatial centering on a frozen local reliability checkpoint."""
import argparse
from pathlib import Path
from shutil import copy2

import torch

from eval_local_reliability_v8 import sensor_probe
from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE
from eval_shared_raw_v8 import shared_raw
from train_videomae import sha, write


NAME = 'local-quality-centered'
ARMS = {NAME: 'videomae2_localcontrast_v8'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', required=True, type=Path)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, output = args.runs.resolve(), args.output.resolve()
    folder = runs/NAME
    folder.mkdir(parents=True, exist_ok=True)
    # Reuse the trained state unchanged. The architecture selects the different gate.
    source_hash = sha(args.initial)
    target = folder/'best_ema.pt'
    if not target.is_file():
        temporary = folder/'best_ema.partial'
        copy2(args.initial, temporary)
        temporary.replace(target)
        write(folder/'config.json', {'source_sha256': source_hash, 'spatial_margin': .05})
        write(folder/'summary.json', {'mode': NAME, 'training_steps': 0})
    if sha(target) != source_hash:
        raise ValueError('Spatial centering expects unchanged local quality tensors')
    manifest = read(RAW/'manifest.json')
    shared_raw(runs, output, RAW, manifest, ARMS, BASE)
    compare(runs, output, manifest, arms=ARMS, columns=('逐帧空间居中',),
            title='局部可靠度按传感器居中', baseline=BASE)
    sensor_probe(runs, manifest, variants={NAME: 'localcontrast'})


if __name__ == '__main__':
    main()
