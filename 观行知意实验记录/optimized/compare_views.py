"""用已验证的缓存快速比较同一模型的原始/翻转视角，之后再独立读取图片复验。"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_generalize import Clips, RawModel, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    config = json.loads((a.run/'config.json').read_text(encoding='utf-8'))
    cache = config.get('val_cache') or config['cache']
    cached_frames = json.loads((Path(cache)/'manifest.json').read_text(encoding='utf-8'))['shape_per_sample'][1]
    ds = Clips(cache, 'validation', frames=config.get('input_frames', cached_frames))
    loader = DataLoader(ds, batch_size=config['batch'], shuffle=False, num_workers=0, pin_memory=True)
    if config.get('architecture') == 'swin3dt':
        from video_swin import VideoSwin
        model = VideoSwin(dropout=.5)
    else:
        model = RawModel(channels=4, dropout=.5)
    model.load_state_dict(torch.load(a.run/'best_ema.pt', map_location='cpu', weights_only=True), strict=True)
    model.to(device='cuda', memory_format=torch.channels_last_3d)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    results = []
    for flip in [False, True]:
        m, probs, labels = evaluate(model, loader, flip=flip)
        mode = 'flip' if flip else 'single'
        np.savez_compressed(a.run/f'cached-{mode}-predictions.npz', probabilities=probs, labels=labels,
                            sample_ids=np.array([r['sample_id'] for r in ds.rows]))
        results.append({'run': a.run.as_posix(), 'flip_tta': flip, 'accuracy': m['accuracy'], 'macro_f1': m['macro_f1']})
        print(json.dumps(results[-1]), flush=True)
    (a.run/'cached-view-comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
