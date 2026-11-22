"""Compare competitor-focused non-target distillation on original validation clips."""
import argparse
from pathlib import Path

import torch

from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE
from soup_domain_v8 import average_states
from train_videomae import save, sha, write


ARMS = dict.fromkeys(('top-5', 'top-12', 'top-average', 'top-5-scaled'), 'videomae2_dual_v8')
COLUMNS = ('保留前 5 类', '保留前 12 类', '两组权重平均', '前 5 类提高蒸馏系数')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--only', choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs = args.runs.resolve()
    folder = runs/'top-average'
    if args.only in (None, 'top-average') and not (folder/'summary.json').is_file():
        sources = [runs/name/'best_ema.pt' for name in ('top-5', 'top-12')]
        states = [torch.load(path, weights_only=True, map_location='cpu') for path in sources]
        folder.mkdir(exist_ok=True)
        save(folder/'best_ema.pt', average_states(states))
        write(folder/'config.json', {'method': 'equal parameter averaging', 'components': ['top-5', 'top-12'],
                                     'source_sha256': [sha(path) for path in sources]})
        write(folder/'summary.json', {'mode': 'top-average', 'best_epoch': None})
    arms = {args.only: ARMS[args.only]} if args.only else ARMS
    compare(runs, args.output.resolve(), read(RAW/'manifest.json'), arms=arms, columns=COLUMNS,
            title='聚焦竞争类别的非目标蒸馏', baseline=BASE, report=args.only is None)


if __name__ == '__main__':
    main()
