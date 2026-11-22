"""Evaluate training-time modality dropout with the usual dual inference path."""
import argparse
from pathlib import Path

import torch

from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE, INIT
from eval_shared_raw_v8 import shared_raw
from fit_frozen_head_v8 import official_split
from probe_modality_experts_v8 import VARIANTS
from cache_teacher_v8 import training_logits
from train_videomae import metrics, write
from videomae_v8 import DomainVideoMAE


ARMS = dict.fromkeys(('adapter-control','adapter-moddrop'),'videomae2_dual_v8')
COLUMNS = ('原适配器对照','训练时随机隐去一路')


def modality_probe(runs, arms):
    data, _, labels = official_split('validation', 466)
    sources = {'initial':INIT, **{name:runs/name/'best_ema.pt' for name in arms}}
    modes = ('dual','depth-only','ir-only')
    results = {}
    for name,path in sources.items():
        row = {}
        for mode, variant in zip(modes,VARIANTS):
            logits = training_logits(DomainVideoMAE(variant=variant),data,path)
            score = metrics(labels,logits)
            score.pop('confusion')
            row[mode] = score
        results[name] = row
        print(name,row,flush=True)
    write(runs/'modality-probe.json',{'samples':len(labels),'results':results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True,type=Path)
    parser.add_argument('--output', required=True,type=Path)
    parser.add_argument('--low', action='store_true', help='Evaluate the 10% dropout experiment')
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs,out = args.runs.resolve(),args.output.resolve()
    manifest = read(RAW/'manifest.json')
    arms = {'adapter-moddrop-10':'videomae2_dual_v8'} if args.low else ARMS
    columns = ('10% 随机隐藏',) if args.low else COLUMNS
    shared_raw(runs,out,RAW,manifest,arms,BASE)
    compare(runs,out,manifest,arms=arms,columns=columns,title='训练时随机隐藏整路模态',baseline=BASE)
    modality_probe(runs, arms)


if __name__ == '__main__':
    main()
