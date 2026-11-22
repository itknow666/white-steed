"""Evaluate learned spatial fusion directions and last-block fine-tuning."""
import argparse
from pathlib import Path

import torch

from eval_local_reliability_v8 import sensor_probe
from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE
from eval_shared_raw_v8 import shared_raw


ARMS = dict.fromkeys(('centered-ce', 'centered-direction', 'centered-top2-ce', 'centered-top2-direction'),
                     'videomae2_localcontrast_v8')
COLUMNS = ('冻结骨干与分类损失', '冻结骨干与配比监督', '末两块与分类损失', '末两块与配比监督')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, out = args.runs.resolve(), args.output.resolve()
    manifest = read(RAW/'manifest.json')
    shared_raw(runs, out, RAW, manifest, ARMS, BASE)
    compare(runs, out, manifest, arms=ARMS, columns=COLUMNS, title='局部配比方向监督与末层微调', baseline=BASE)
    sensor_probe(runs, manifest, variants=dict.fromkeys(ARMS, 'localcontrast'))


if __name__ == '__main__':
    main()
