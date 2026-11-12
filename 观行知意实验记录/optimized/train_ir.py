"""优化实验：温和类别加权、分层学习率、慢BN统计、EMA和可恢复训练。

仅使用已发布训练/验证划分；验证集不参与梯度、类别权重或BN统计计算。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from timm.utils import ModelEmaV3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baseline' if (ROOT / 'baseline').is_dir() else ROOT))
from raw_model import RawModel


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    temp = path.with_suffix('.partial')
    torch.save(data, temp)
    temp.replace(path)


class Clips(Dataset):
    def __init__(self, cache, split, training=False, frames=None, temporal_jitter=False, erase_prob=0.):
        self.cache = Path(cache)
        self.meta = json.loads((self.cache / 'manifest.json').read_text(encoding='utf-8'))
        self.rows = sorted([r for r in self.meta['samples'] if r['split'] == split], key=lambda r: r['row'])
        self.split, self.training, self.array = split, training, None
        self.frames, self.temporal_jitter, self.erase_prob = frames, temporal_jitter, erase_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.cache / (self.split + '.npy'), mmap_mode='r')
        x = torch.from_numpy(self.array[index].copy()).float().div_(255)
        if self.frames and self.frames != x.shape[1]:
            if self.frames > x.shape[1]:
                raise ValueError('Requested more frames than cached')
            if self.training and self.temporal_jitter:
                # 各时间段随机选一帧，两个模态共享时间索引；不反转动作顺序。
                edges = np.linspace(0, x.shape[1], self.frames+1).astype(int)
                ids = [random.randrange(edges[i], edges[i+1]) for i in range(self.frames)]
            else:
                ids = np.linspace(0, x.shape[1]-1, self.frames).astype(int).tolist()
            x = x[:, ids]
        if self.training:
            if random.random() < .5:
                x = x.flip(-1)
            # 比旧版0.75的下限更温和；整段、四个通道共享几何变换。
            size = x.shape[-1]
            side = round(size * random.uniform(.85, 1.0))
            top, left = random.randint(0, size - side), random.randint(0, size - side)
            x = F.interpolate(x.permute(1, 0, 2, 3)[:, :, top:top+side, left:left+side],
                              size=(size, size), mode='bilinear', align_corners=False).permute(1, 0, 2, 3)
            if random.random() < self.erase_prob:
                # 同步小区域遮挡，模拟遮挡；不对伪彩深度做RGB色相变换。
                side = max(1, round(size*math.sqrt(random.uniform(.02, .08))))
                top, left = random.randint(0, size-side), random.randint(0, size-side)
                x[:, :, top:top+side, left:left+side] = 0.
        return x, int(self.rows[index]['label']), index


def metrics(labels, probabilities):
    if not np.isfinite(probabilities).all():
        raise RuntimeError('Nonfinite validation probabilities')
    cm = np.bincount(labels * 40 + probabilities.argmax(1), minlength=1600).reshape(40, 40)
    tp = cm.diagonal()
    return {'accuracy': float(100 * tp.sum() / len(labels)),
            'macro_f1': float(100 * np.mean(2 * tp / np.maximum(1, cm.sum(0) + cm.sum(1)))),
            'confusion': cm.tolist()}


@torch.inference_mode()
def evaluate(model, loader, flip=False):
    model.eval()
    probs = np.empty((len(loader.dataset), 40), dtype=np.float32)
    labels = np.empty(len(loader.dataset), dtype=np.int64)
    for x, y, index in loader:
        x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(x).float().softmax(1)
            if flip:
                p = (p + model(x.flip(-1)).float().softmax(1)) / 2
        probs[index.numpy()], labels[index.numpy()] = p.cpu().numpy(), y.numpy()
    return metrics(labels, probs), probs, labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial', type=Path, default=ROOT / 'runs/initial16-20261020-5ep/best_ema.pt')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--head-multiplier', type=float, default=10)
    p.add_argument('--bn-momentum', type=float, default=.01)
    p.add_argument('--class-power', type=float, default=.25)
    p.add_argument('--smoothing', type=float, default=.05)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--val-cache', type=Path)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--freeze-bn', action='store_true', help='冻结运行统计，BN仿射参数继续微调')
    p.add_argument('--temporal-jitter', action='store_true')
    p.add_argument('--mixup-alpha', type=float, default=0.)
    p.add_argument('--mixup-prob', type=float, default=.5)
    p.add_argument('--erase-prob', type=float, default=0.)
    p.add_argument('--min-epochs', type=int, default=4)
    a = p.parse_args()
    if a.output.exists() and not a.resume:
        raise FileExistsError(f'Preserve previous run: {a.output}')
    a.output.mkdir(parents=True, exist_ok=a.resume)
    torch.set_num_threads(4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train = Clips(a.cache, 'train', True, a.frames, a.temporal_jitter, a.erase_prob)
    val = Clips(a.val_cache or a.cache, 'validation', frames=a.frames)
    if train.meta['shape_per_sample'][2:] != val.meta['shape_per_sample'][2:]:
        raise ValueError('Training/validation cache spatial dimensions differ')
    if a.val_cache:
        expected = {(r['sample_id'], r['label']) for r in train.meta['samples'] if r['split']=='validation'}
        actual = {(r['sample_id'], r['label']) for r in val.rows}
        if expected != actual:
            raise ValueError('Separate validation cache IDs or labels differ')
    if len(train) != 1995 or len(val) != 466 or train.meta.get('smoke'):
        raise ValueError('Expected full published training and validation sets')
    subjects = [{r['subject'] for r in ds.rows} for ds in [train, val]]
    if subjects[0] & subjects[1]:
        raise ValueError('Subjects overlap')
    counts = np.bincount([r['label'] for r in train.rows], minlength=40)
    if np.any(counts == 0):
        raise ValueError('Training is missing a class')
    weights = counts.astype(np.float32) ** -a.class_power
    weights /= weights.mean()
    model = RawModel(channels=4, pretrained=None, dropout=.5)
    model.load_state_dict(torch.load(a.initial, map_location='cpu', weights_only=True), strict=True)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.momentum = a.bn_momentum
    model.to(device='cuda', memory_format=torch.channels_last_3d)
    params = sum(q.numel() for q in model.parameters()) + train.meta['detector_parameters']
    if params > 100_000_000:
        raise ValueError('Inference pipeline exceeds 100M')
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': a.lr, 'lr_scale': 1.0},
        {'params': model.head.parameters(), 'lr': a.lr*a.head_multiplier, 'lr_scale': a.head_multiplier},
    ], weight_decay=.01)
    ema = ModelEmaV3(model, decay=.99, device='cuda')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device='cuda'), label_smoothing=a.smoothing)
    train_loader = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=a.batch, shuffle=False, num_workers=0, pin_memory=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(amp='bfloat16', channels_last_3d=True, bn_momentum=a.bn_momentum, dropout=.5, weight_decay=.01,
                  class_counts_train_only=counts.tolist(), class_weights_train_only=weights.tolist(),
                  cache_sha256=sha(a.cache/'manifest.json'), initial_sha256=sha(a.initial),
                  total_parameters=params, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  train_samples=1995, validation_samples=466, person_crop=train.meta['person_crop'],
                  input_frames=a.frames, training_source='optimized/train_ir.py', ir_contrast=train.meta.get('ir_contrast',False),
                  val_cache_sha256=sha((a.val_cache or a.cache)/'manifest.json'),
                  training_recipe='v3_frozen_bn_temporal_or_resolution_regularization',
                  selection='max validation accuracy, then macro_f1, then earlier epoch', test_accessed=False)
    start_epoch, step, best, best_epoch, stale = 1, 0, (-1., -1.), 0, 0
    if a.resume:
        checkpoint = torch.load(a.output/'resume.pt', map_location='cpu', weights_only=False)
        prior = json.loads((a.output/'config.json').read_text(encoding='utf-8'))
        for key in ['epochs', 'batch', 'lr', 'head_multiplier', 'bn_momentum', 'class_power', 'smoothing', 'cache_sha256', 'initial_sha256', 'val_cache_sha256', 'freeze_bn', 'frames', 'temporal_jitter', 'mixup_alpha', 'mixup_prob', 'erase_prob', 'min_epochs']:
            if prior[key] != config[key]:
                raise ValueError(f'Resume config differs: {key}')
        model.load_state_dict(checkpoint['model'])
        ema.module.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        start_epoch, step, best, best_epoch, stale = checkpoint['epoch']+1, checkpoint['step'], checkpoint['best'], checkpoint['best_epoch'], checkpoint['stale']
    else:
        write(a.output/'config.json', config)
        write(a.output/'metadata.json', {'smoke': False, 'parameters': params,
              'recipe': config['training_recipe'], 'initial_sha256': config['initial_sha256'],
              'test_accessed': False, 'person_crop': train.meta['person_crop']})
    total_steps = len(train_loader)*a.epochs
    warmup = len(train_loader)  # 一个epoch温和升温；每次续训保留原计划。
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, a.epochs+1):
        model.train()
        if a.freeze_bn:
            # model.train()每轮都会打开BN，因此每轮重新固定训练集已学到的统计。
            for module in model.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
        losses, seen, correct, epoch_start = 0., 0, 0, time.time()
        for batch_index, (x, y, _) in enumerate(train_loader, 1):
            factor = .2 + .8*(step+1)/warmup if step < warmup else .05 + .95*.5*(1+math.cos(math.pi*(step-warmup)/max(1, total_steps-warmup)))
            for group in optimizer.param_groups:
                group['lr'] = a.lr*group['lr_scale']*factor
            x, y = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last_3d), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            lam, permutation = 1., None
            if a.mixup_alpha > 0 and len(y) > 1 and random.random() < a.mixup_prob:
                lam = float(np.random.beta(a.mixup_alpha, a.mixup_alpha))
                permutation = torch.randperm(len(y), device=y.device)
                # 整个片段与两个模态使用相同混合比例；只混合训练样本。
                x = (lam*x + (1-lam)*x[permutation]).contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scores = model(x)
                loss = criterion(scores, y) if permutation is None else lam*criterion(scores,y)+(1-lam)*criterion(scores,y[permutation])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            ema.update(model, step=step)
            step += 1
            losses += loss.item()*len(y)
            correct += float((scores.argmax(1) == y).sum())*lam
            if permutation is not None:
                correct += float((scores.argmax(1) == y[permutation]).sum())*(1-lam)
            seen += len(y)
            if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(train_loader):
                progress = {'epoch': epoch, 'epochs': a.epochs, 'batch': batch_index, 'batches': len(train_loader),
                            'seen': seen, 'train_loss': losses/seen, 'train_accuracy': 100*correct/seen,
                            'elapsed_sec': time.time()-started, 'lr': optimizer.param_groups[0]['lr'],
                            'grad_norm': float(norm), 'peak_gpu_mib': torch.cuda.max_memory_allocated()/2**20}
                write(a.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if seen != 1995:
            raise RuntimeError('Not all training samples visited')
        train_seconds = time.time()-epoch_start
        m, probabilities, labels = evaluate(ema.module, val_loader)
        score = (m['accuracy'], m['macro_f1'])
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            save(a.output/'best_ema.pt', ema.module.state_dict())
            write(a.output/'best_metrics.json', dict(m, epoch=epoch))
            np.savez_compressed(a.output/'best_predictions.npz', probabilities=probabilities, labels=labels,
                                sample_ids=np.array([r['sample_id'] for r in val.rows]))
        else:
            stale += 1
        save(a.output/'last_ema.pt', ema.module.state_dict())
        row = {'epoch': epoch, 'train_loss': losses/seen, 'train_accuracy': 100*correct/seen,
               'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'], 'best_accuracy': best[0],
               'best_epoch': best_epoch, 'train_samples_seen': seen, 'steps': step, 'train_sec': train_seconds,
               'elapsed_sec': time.time()-started, 'lr_backbone': optimizer.param_groups[0]['lr']}
        with (a.output/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        save(a.output/'resume.pt', {'model': model.state_dict(), 'ema': ema.module.state_dict(),
             'optimizer': optimizer.state_dict(), 'epoch': epoch, 'step': step,
             'best': best, 'best_epoch': best_epoch, 'stale': stale,
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
        print(json.dumps(row), flush=True)
        if epoch >= a.min_epochs and stale >= a.patience:
            break
    best_metrics = json.loads((a.output/'best_metrics.json').read_text(encoding='utf-8'))
    write(a.output/'summary.json', {'best_epoch': best_epoch, 'epochs_completed': epoch,
          'accuracy': best_metrics['accuracy'], 'macro_f1': best_metrics['macro_f1'],
          'steps': step, 'parameters': params, 'elapsed_sec': time.time()-started,
          'peak_gpu_mib': torch.cuda.max_memory_allocated()/2**20,
          'stopped_early': epoch < a.epochs, 'linux_verified': False})


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
