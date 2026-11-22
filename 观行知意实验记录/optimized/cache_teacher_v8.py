"""Save local teacher logits for the published training split only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from package_domain_v8 import R34, ROOT
from train_videomae import Clips, RawModel, sha, write
from videomae_v8 import DomainVideoMAE


VIDEO_SOURCE = ROOT/'runs/v8-paper-sprint-20261122/soup-three/best_ema.pt'
R34_SOURCE = R34/'best_ema.pt'


@torch.inference_mode()
def training_logits(model, data, weights):
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.cuda().to(memory_format=torch.channels_last_3d).eval()
    out = np.empty((len(data), 40), dtype=np.float32)
    for x, _, indices in DataLoader(data, batch_size=8, pin_memory=True):
        x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x).float()
        out[indices.numpy()] = logits.cpu().numpy()
    print(weights.parent.name, len(out), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    videos = Clips(ROOT/'cache/crop16-192-v3', 'train', frames=16)
    r34 = Clips(ROOT/'cache/crop16-v2', 'train', frames=16)
    release = json.loads((ROOT/'data/raw/train/manifest.json').read_text(encoding='utf-8'))
    truth = {row.get('sample_id', row['file']+'.npy'): int(row['label']) for row in release['samples']}
    video_ids = [row['sample_id'] for row in videos.rows]
    r34_ids = {row['sample_id']: i for i, row in enumerate(r34.rows)}
    if len(truth) != 1995 or len(video_ids) != 1995 or len(r34.rows) != 1995 or len(set(video_ids)) != 1995 or set(video_ids) != set(r34_ids) or set(video_ids) != set(truth):
        raise ValueError('Both caches must match the complete official training split')
    reorder = [r34_ids[k] for k in video_ids]
    labels = [truth[k] for k in video_ids]
    if labels != [row['label'] for row in videos.rows] or labels != [r34.rows[i]['label'] for i in reorder]:
        raise ValueError('Training labels disagree across the source caches')
    video_logits = training_logits(DomainVideoMAE(), videos, VIDEO_SOURCE)
    r34_logits = training_logits(RawModel(channels=4, pretrained=None), r34, R34_SOURCE)[reorder]
    if not np.isfinite(video_logits).all() or not np.isfinite(r34_logits).all():
        raise ValueError('Nonfinite teacher logits')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial')
    with temporary.open('wb') as f:
        np.savez_compressed(f, sample_ids=video_ids, labels=labels, video_logits=video_logits, r34_logits=r34_logits)
    temporary.replace(args.output)
    write(args.output.with_suffix('.json'), {'split': 'train', 'samples': len(labels),
                                              'r34_sha256': sha(R34_SOURCE), 'video_sha256': sha(VIDEO_SOURCE),
                                              'train_manifest_sha256': sha(ROOT/'data/raw/train/manifest.json'),
                                              'teacher_cache_sha256': sha(args.output)})


if __name__ == '__main__':
    main()
