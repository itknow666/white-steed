"""Compare teacher-transfer and frozen-adapter experiments on raw clips."""
import argparse
from pathlib import Path

import torch

from eval_phase_order_v8 import RAW, compare, read


ARMS = dict.fromkeys(('control', 'kd-fusion', 'kd-adaptive', 'adapter-control', 'adapter-kd', 'kd-non-target',
                      'kd-non-target-seed43'),
                   'videomae2_dual_v8')
COLUMNS = ('普通续训', '固定融合蒸馏', '按样本蒸馏', '仅适配器', '仅适配器加蒸馏', '蒸馏其他类别', '非目标蒸馏种子 43')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', choices=ARMS, help='Verify one completed arm before writing the full report')
    args = parser.parse_args()
    torch.set_num_threads(4)
    arms = {args.only: ARMS[args.only]} if args.only else ARMS
    compare(args.runs, args.output, read(RAW/'manifest.json'), arms=arms, columns=COLUMNS,
            title='融合教师迁移', training_sources=('cache_teacher_v8.py',), report=args.only is None)


if __name__ == '__main__':
    main()
