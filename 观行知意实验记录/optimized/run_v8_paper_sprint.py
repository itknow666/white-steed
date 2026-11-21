"""Train one lane of the direction-A paper sprint in a fresh child per arm."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from train_videomae import write


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'runs/v8-paper-sprint-20261122'
BASE = ROOT/'runs/v8-domain-20261122/dual/best_ema.pt'
LANES = {
    'a': ('transfer43', [('plain', []), ('temporal', ['--domain-variant','temporal']),
                         ('distill', ['--domain-variant','gate','--mask-rate','.15','--distill','.5'])]),
    'b': ('transfer44', [('gate', ['--domain-variant','gate']),
                         ('masked', ['--domain-variant','gate','--mask-rate','.15'])]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('lane', choices=LANES)
    p.add_argument('--cutoff', required=True, help='UTC ISO 8601 deadline for training')
    a = p.parse_args()
    cutoff = datetime.fromisoformat(a.cutoff).astimezone(timezone.utc)
    initial, trials = LANES[a.lane]
    while not (RUNS/initial/'summary.json').is_file():
        if datetime.now(timezone.utc) >= cutoff:
            raise TimeoutError(initial)
        time.sleep(5)
    outcomes = {}
    for name, flags in trials:
        seconds = (cutoff - datetime.now(timezone.utc)).total_seconds()
        if seconds < 35:
            outcomes[name] = 'budget'
            continue
        args = [sys.executable, '-u', str(ROOT/'optimized/train_domain_v8.py'),
                '--cache', str(ROOT/'cache/crop16-192-v3'), '--initial', str(BASE),
                '--output', str(RUNS/name), '--mode','dual','--epochs','2','--seed','42',
                '--base-lr','1e-5','--continue-dual', *flags]
        with (RUNS/(name+'.log')).open('x', encoding='utf-8') as log:
            print('start', name, flush=True)
            try:
                code = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=seconds, check=False).returncode
                outcomes[name] = 'complete' if code == 0 else f'exit-{code}'
            except subprocess.TimeoutExpired:
                outcomes[name] = 'budget'
            print('finish', name, outcomes[name], flush=True)
    write(RUNS/f'lane-{a.lane}.json', outcomes)


if __name__ == '__main__':
    main()
