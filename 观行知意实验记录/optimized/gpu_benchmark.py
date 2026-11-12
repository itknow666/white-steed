"""检验batch8的BF16更新，记录显存与吞吐；不发布模型。"""
from pathlib import Path
import argparse
import json
import sys
import time
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'baseline'))
from raw_model import RawModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--channels-last', action='store_true')
    a = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    model = RawModel(4).cuda().train()
    state = torch.load(ROOT/'runs/initial16-20261020-5ep/best_ema.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(state)
    if a.channels_last:
        model.to(memory_format=torch.channels_last_3d)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.momentum = .01
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    x = torch.rand(a.batch, 4, 16, 128, 128, device='cuda')
    y = torch.arange(a.batch, device='cuda') % 40
    if a.channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for step in range(4):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if step:
            times.append(time.perf_counter()-start)
        losses.append(float(loss.detach()))
    result = {'batch': a.batch, 'amp': 'bfloat16', 'channels_last_3d': a.channels_last, 'finite_gradients': bool(torch.isfinite(norm)),
              'peak_gpu_mib_without_ema': torch.cuda.max_memory_allocated()/2**20,
              'step_seconds_mean_after_warmup': sum(times)/len(times), 'losses': losses,
              'synthetic_inputs_for_performance_only': True, 'published_weights': False}
    suffix = '-channels-last' if a.channels_last else ''
    (ROOT/f'runs/profile-v2-batch{a.batch}{suffix}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
