"""Validate anchored continuation, patch repair, and fixed checkpoint mixtures."""
import argparse
from pathlib import Path

import torch

from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE, INIT, synthetic_probe
from package_domain_v8 import ROOT
from soup_domain_v8 import average_states
from train_videomae import save, sha, write


TRAINED = ('anchor-clean', 'anchor-depth', 'patch-repair', 'anchor-replay', 'patch-only', 'replay-no-drop')
MIXTURES = {'half-depth': (None, 'anchor-depth'), 'half-repair': (None, 'patch-repair'),
            'repair-replay': ('patch-repair', 'anchor-replay'), 'clean-depth-mixture': ('replay-no-drop', 'half-depth')}
ARMS = dict.fromkeys((*TRAINED, *MIXTURES), 'videomae2_dual_v8')
NAMES = dict(zip(ARMS, ('起点锚定', '深度缺损锚定', '局部特征修复', '锚定加非目标蒸馏', '仅输入层修复', '非目标蒸馏关闭随机丢弃',
                       '深度锚定与起点平均', '修复与起点平均', '修复与非目标平均', '关闭随机丢弃与折中权重平均')))
CASES = ('clean', 'dark-ir', 'depth-gap', 'short-offset-gap')


def make_mixtures(runs):
    for name, source_names in MIXTURES.items():
        if any(n is not None and not (runs/n/'summary.json').is_file() for n in source_names):
            continue
        folder = runs/name
        if (folder/'summary.json').is_file():
            continue
        sources = [INIT if n is None else runs/n/'best_ema.pt' for n in source_names]
        states = [torch.load(path, map_location='cpu', weights_only=True) for path in sources]
        folder.mkdir(exist_ok=True)
        save(folder/'best_ema.pt', average_states(states))
        write(folder/'config.json', {'method': 'equal parameter averaging',
                                     'sources': [str(p.relative_to(ROOT)) for p in sources],
                                     'source_sha256': [sha(p) for p in sources]})
        write(folder/'summary.json', {'mode': name, 'best_epoch': None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--prepare-mixtures', action='store_true')
    operation.add_argument('--probe-only', action='store_true')
    operation.add_argument('--report-only', action='store_true')
    operation.add_argument('--only', choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runs, out = args.runs.resolve(), args.output.resolve()
    if args.prepare_mixtures:
        make_mixtures(runs)
        return
    if args.probe_only:
        arms = [name for name in ARMS if (runs/name/'summary.json').is_file()]
        path = runs/'synthetic-stress.json'
        previous = read(path) if path.is_file() else {}
        cached = previous.get('results', {}) if previous.get('cases') == list(CASES) else {}
        pending = [name for name in arms if name not in cached]
        if pending or 'initial' not in cached:
            result = synthetic_probe(runs, arms=pending, cases=CASES)
            result['results'] = {**cached, **result['results']}
            write(path, result)
        return
    if args.only:
        names = [args.only]
    elif args.report_only:
        names = [name for name in ARMS if (out/name/'metrics.json').is_file()]
    else:
        make_mixtures(runs)
        names = [name for name in ARMS if (runs/name/'summary.json').is_file()]
    arms = {name: ARMS[name] for name in names}
    compare(runs, out, read(RAW/'manifest.json'), arms=arms, columns=[NAMES[n] for n in names],
            title='干净教师锚定与缺损特征修复', baseline=BASE, report=args.only is None)
    if not args.only:
        write(runs/'evaluation-status.json', {'evaluated': names, 'other_candidates': [n for n in ARMS if n not in names]})


if __name__ == '__main__':
    main()
