"""Controlled V7 warm-start comparison on the official training split."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from timm.utils import ModelEmaV3
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_videomae import Clips, evaluate, save, sha, write
from videomae_model import VideoMAEClassifier
from videomae_v8 import DomainVideoMAE


def adapt_without_labels(model, loader):
    """Use one modality to predict the other's masked tubelet mean."""
    patch = model.encoder.patch_embed
    projections = list(patch.depth.parameters()) + list(patch.ir.parameters())
    for parameter in projections:
        parameter.requires_grad_(False)
    heads = nn.ModuleList([nn.Linear(384, 1), nn.Linear(384, 3)]).cuda()
    optimizer = torch.optim.AdamW(
        list(patch.depth_adapter.parameters()) + list(patch.ir_adapter.parameters()) +
        list(heads.parameters()), lr=3e-4)
    loss_sum = 0.
    for x, _, _ in loader:
        normalized = model.normalized_clip(x.cuda(non_blocking=True))
        depth = F.avg_pool3d(normalized[:, :3], (2, 16, 16)).flatten(2).transpose(1, 2)
        ir = F.avg_pool3d(normalized[:, 3:], (2, 16, 16)).flatten(2).transpose(1, 2)
        # At each selected tubelet, each head sees only the opposite modality.
        selected = torch.rand(depth.shape[:2], device='cuda') < .25
        selected[:, 0] = True
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            d, i = patch.streams(normalized)
            loss = F.smooth_l1_loss(heads[0](d[selected]), ir[selected])
            loss = loss + F.smooth_l1_loss(heads[1](i[selected]), depth[selected])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite self-supervised loss')
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach())
    for parameter in projections:
        parameter.requires_grad_(True)
    return loss_sum / len(loader)


def make_model(mode, state):
    if mode == 'control':
        model = VideoMAEClassifier()
        model.load_state_dict(state)
    else:
        model = DomainVideoMAE('lora' if mode == 'lora' else 'dual', warmstart=state)
    if mode in ('staged', 'ssl', 'lora'):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('patch_embed' in name or name.startswith('head.') or
                                     'fc_norm' in name or '.attn.down.' in name or '.attn.up.' in name)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True, help='V7 VideoMAE single-model checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['control', 'dual', 'lora', 'staged', 'ssl'], required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=8)
    a = parser.parse_args()
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    train = Clips(a.cache, 'train', True, 16, erase_prob=.10)
    valid = Clips(a.cache, 'validation', frames=16)
    if (len(train), len(valid)) != (1995, 466) or {x['subject'] for x in train.rows} & {x['subject'] for x in valid.rows}:
        raise ValueError('Expected the official subject-disjoint train/validation split')
    if train.meta['shape_per_sample'] != [4, 16, 192, 192]:
        raise ValueError('Expected synchronized four-channel 16×192 cache')
    tr = DataLoader(train, batch_size=a.batch, shuffle=True, pin_memory=True)
    va = DataLoader(valid, batch_size=a.batch, shuffle=False, pin_memory=True)
    state = torch.load(a.initial, map_location='cpu', weights_only=True)
    model = make_model(a.mode, state).cuda()
    del state
    num_params = sum(t.numel() for t in model.parameters())
    initial_trainable = sum(t.numel() for t in model.parameters() if t.requires_grad)
    config = {'mode': a.mode, 'seed': a.seed, 'batch': a.batch, 'epochs': a.epochs,
              'base_lr': 2e-5, 'layer_decay': .85, 'head_multiplier': 5,
              'initial_sha256': sha(a.initial), 'cache_sha256': sha(a.cache/'manifest.json'),
              'parameters': num_params, 'initial_trainable': initial_trainable,
              'train_samples': len(train), 'validation_samples': len(valid), 'test_accessed': False}
    write(a.output/'config.json', config)
    started = time.monotonic()
    if a.mode == 'ssl':
        ssl_loss = adapt_without_labels(model, tr)
        write(a.output/'ssl.json', {'training_clips': len(train), 'loss': ssl_loss, 'labels_used': False})
        print(json.dumps({'ssl_loss': ssl_loss}), flush=True)
        # Repeat the supervised seed so the unlabeled pass does not change shuffling policy.
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
    groups = model.layer_groups(2e-5, .85, 5, .05)
    # Latter four blocks are unfrozen after epoch one; retain their optimizer groups.
    delayed = [q for block in model.encoder.blocks[-4:] for q in block.parameters()] if a.mode in ('staged', 'ssl') else []
    eligible = {id(t) for t in model.parameters() if t.requires_grad} | {id(t) for t in delayed}
    patch_ids = {id(t) for t in model.encoder.patch_embed.parameters()}
    optimizer_groups = []
    for group in groups:
        chosen = [q for q in group['params'] if id(q) in eligible]
        for is_patch in (False, True):
            selected = [q for q in chosen if (id(q) in patch_ids) == is_patch]
            if selected:
                scale = 2.5 if is_patch and a.mode != 'control' else group['lr_scale']
                optimizer_groups.append(dict(group, params=selected, lr_scale=scale))
    optimizer = torch.optim.AdamW(optimizer_groups)
    ema = ModelEmaV3(model, decay=.99, device='cuda')
    counts = np.bincount([r['label'] for r in train.rows], minlength=40)
    if not np.all(counts > 0):
        raise ValueError('A class is absent from training')
    weights = counts.astype(np.float32) ** -.25
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights / weights.mean(), device='cuda'), label_smoothing=.05)
    best, best_epoch, step = (-1., -1.), 0, 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, a.epochs + 1):
        if epoch == 2:
            for q in delayed:
                q.requires_grad_(True)
        model.train()
        seen, loss_sum = 0, 0.
        for x, y, _ in tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            frac = step / max(1, a.epochs * len(tr) - 1)
            schedule = min(1., .2 + 4. * frac) * (.15 + .85 * .5 * (1 + math.cos(math.pi * frac)))
            for group in optimizer.param_groups:
                group['lr'] = 2e-5 * group['lr_scale'] * schedule
            optimizer.zero_grad(set_to_none=True)
            perm = torch.randperm(len(y), device='cuda') if random.random() < .5 else None
            mix = float(np.random.beta(.2, .2)) if perm is not None else 1.
            if perm is not None:
                x = mix * x + (1. - mix) * x[perm]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
                if perm is not None:
                    loss = mix * loss + (1. - mix) * criterion(logits, y[perm])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            ema.update(model, step=step)
            step += 1
            seen += len(y)
            loss_sum += float(loss.detach()) * len(y)
            if step % 80 == 1:
                print(json.dumps({'mode': a.mode, 'epoch': epoch, 'step': step, 'loss': loss_sum / seen}), flush=True)
        result, probabilities, labels = evaluate(ema.module, va)
        score = result['accuracy'], result['macro_f1']
        if score > best:
            best, best_epoch = score, epoch
            save(a.output/'best_ema.pt', ema.module.state_dict())
            np.savez_compressed(a.output/'best_predictions.npz', probabilities=probabilities, labels=labels,
                                sample_ids=np.asarray([r['sample_id'] for r in valid.rows]))
        row = dict(epoch=epoch, steps=step, seen=seen, loss=loss_sum/seen, accuracy=score[0], macro_f1=score[1],
                   trainable=sum(t.numel() for t in model.parameters() if t.requires_grad),
                   peak_mib=torch.cuda.max_memory_allocated()/2**20, elapsed=time.monotonic()-started)
        with (a.output/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
    write(a.output/'summary.json', dict(mode=a.mode, best_epoch=best_epoch, accuracy=best[0], macro_f1=best[1]))


if __name__ == '__main__':
    main()
