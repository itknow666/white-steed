"""Compare matched dual-view and IR-view replay during recovery."""
import argparse
from pathlib import Path

import torch

from eval_modality_dropout_v8 import modality_probe
from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE
from eval_shared_raw_v8 import shared_raw


ARMS = dict.fromkeys(('recovery-control','clean-replay','ir-replay','both-replay'),'videomae2_dual_v8')
COLUMNS = ('普通双路恢复','双路回放','红外回放','双路和红外回放')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    runs,out=args.runs.resolve(),args.output.resolve()
    manifest=read(RAW/'manifest.json')
    shared_raw(runs,out,RAW,manifest,ARMS,BASE)
    compare(runs,out,manifest,arms=ARMS,columns=COLUMNS,title='双路与红外输出回放',baseline=BASE)
    modality_probe(runs,ARMS)


if __name__=='__main__':
    main()
