"""Compare cached and online teacher views on original validation clips."""
import argparse
from pathlib import Path

import torch

from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE
from soup_domain_v8 import average_states
from train_videomae import save, sha, write


ARMS = dict.fromkeys(('offline-control', 'augmented', 'mixed', 'view-average'), 'videomae2_dual_v8')
COLUMNS = ('离线复跑', '同步裁剪与翻转', '同步 Mixup', '两组在线权重平均')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs = args.runs.resolve()
    folder = runs/'view-average'
    if not (folder/'summary.json').is_file():
        sources = [runs/name/'best_ema.pt' for name in ('augmented', 'mixed')]
        states = [torch.load(path, weights_only=True, map_location='cpu') for path in sources]
        folder.mkdir(exist_ok=True)
        save(folder/'best_ema.pt', average_states(states))
        write(folder/'config.json', {'method': 'equal parameter averaging', 'sources': ['augmented', 'mixed'],
                                     'source_sha256': [sha(path) for path in sources]})
        write(folder/'summary.json', {'mode': 'view-average', 'best_epoch': None})
    compare(runs, args.output.resolve(), read(RAW/'manifest.json'), arms=ARMS, columns=COLUMNS,
            title='同视图非目标蒸馏', baseline=BASE)


if __name__ == '__main__':
    main()
