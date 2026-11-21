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


def local_alignment(x, depth, ir, sampling, rng):
    """Match synchronized depth and IR tokens against distant tokens in each clip."""
    with torch.no_grad():
        if sampling in ('motion', 'motion-stratified'):
            change = x.diff(dim=2).abs()
            change = torch.cat((change[:, :, :1], change), dim=2)
            depth_change = F.adaptive_avg_pool3d(change[:, :3].mean(dim=1, keepdim=True), (8, 14, 14)).flatten(1)
            ir_change = F.adaptive_avg_pool3d(change[:, 3:4], (8, 14, 14)).flatten(1)
            # Give both modalities equal weight even if their raw contrast differs.
            depth_change /= depth_change.mean(1, keepdim=True).clamp_min(1e-5)
            ir_change /= ir_change.mean(1, keepdim=True).clamp_min(1e-5)
            weights = depth_change * ir_change
        else:
            weights = torch.rand(depth.shape[:2], device=x.device, generator=rng)
        if sampling == 'motion-stratified':
            per_time = weights.reshape(-1, 8, 196).topk(4, dim=2).indices
            offsets = torch.arange(8, device=x.device).view(1, 8, 1) * 196
            selected = (per_time + offsets).flatten(1)
        else:
            selected = weights.topk(32, dim=1).indices
        time, row, column = selected // 196, selected // 14 % 14, selected % 14
        close = (time[:, :, None] - time[:, None, :]).abs() <= 1
        close &= (row[:, :, None] - row[:, None, :]).abs() <= 1
        close &= (column[:, :, None] - column[:, None, :]).abs() <= 1
        label = torch.arange(selected.shape[1], device=x.device)
        close &= ~torch.eye(selected.shape[1], device=x.device, dtype=torch.bool)
    indices = selected.unsqueeze(-1).expand(-1, -1, depth.size(-1))
    depth_tokens = depth.gather(1, indices).float()
    ir_tokens = ir.gather(1, indices).float()
    depth_tokens = F.normalize(depth_tokens - depth_tokens.mean(-1, keepdim=True), dim=-1, eps=1e-6)
    ir_tokens = F.normalize(ir_tokens - ir_tokens.mean(-1, keepdim=True), dim=-1, eps=1e-6)
    similarity = (depth_tokens @ ir_tokens.transpose(1, 2)).float() / .1
    similarity = similarity.masked_fill(close, -1e4)
    target = label.expand(depth.size(0), -1)
    return (F.cross_entropy(similarity, target) + F.cross_entropy(similarity.transpose(1, 2), target)) / 2


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
    parser.add_argument('--continue-dual', action='store_true', help='Initialize from a trained V8 dual checkpoint')
    parser.add_argument('--base-lr', type=float, default=2e-5)
    parser.add_argument('--keep-blocks-frozen', action='store_true', help='Keep staged Transformer blocks frozen for every epoch')
    parser.add_argument('--class-power', type=float, default=.25)
    parser.add_argument('--ir-jitter', action='store_true', help='Training-only clip-consistent IR gain and offset')
    parser.add_argument('--domain-variant', choices=['plain', 'gate', 'temporal', 'phase'], default='plain')
    parser.add_argument('--mask-rate', type=float, default=0.)
    parser.add_argument('--distill', type=float, default=0., help='KL weight for frozen clean-view V8 teacher')
    parser.add_argument('--alignment', choices=['none', 'control', 'uniform', 'motion', 'motion-stratified'], default='none')
    parser.add_argument('--order-weight', type=float, default=0., help='Auxiliary before/after prediction weight')
    parser.add_argument('--phase-head-only', action='store_true', help='Freeze the starting model and train only the phase residual')
    parser.add_argument('--pace-jitter', action='store_true', help='Synchronized monotonic time warp during training')
    a = parser.parse_args()
    if a.continue_dual and a.mode not in ('dual', 'staged'):
        parser.error('--continue-dual supports dual and staged')
    if a.base_lr <= 0 or a.class_power < 0:
        parser.error('Expected positive learning rate and nonnegative class exponent')
    if a.keep_blocks_frozen and a.mode != 'staged':
        parser.error('--keep-blocks-frozen requires staged mode')
    if a.domain_variant != 'plain' and not a.continue_dual:
        parser.error('New domain variants require a V8 dual checkpoint')
    if a.alignment != 'none' and (a.mode != 'dual' or not a.continue_dual or a.domain_variant != 'plain' or a.mask_rate):
        parser.error('Local alignment requires an unmasked plain dual checkpoint')
    if a.order_weight < 0 or (a.order_weight and (a.domain_variant != 'phase' or a.mask_rate)):
        parser.error('Auxiliary direction prediction requires an unmasked phase head')
    if a.phase_head_only and (a.mode != 'dual' or a.domain_variant != 'phase' or a.order_weight):
        parser.error('Phase-only fitting requires phase classification without auxiliary direction prediction')
    if not 0 <= a.mask_rate < 1 or a.distill < 0 or (a.distill and not (a.continue_dual and a.mask_rate)):
        parser.error('Distillation requires a V8 checkpoint and a positive mask rate')
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
    if a.continue_dual:
        model = DomainVideoMAE(variant=a.domain_variant)
        missing, extra = model.load_state_dict(state, strict=False)
        new_keys = {key for key in model.state_dict() if any(
            part in key for part in ('sensor_gate', 'temporal_adapter', 'phase_head'))}
        if set(missing) != new_keys or extra:
            raise ValueError(f'Unexpected V8 migration keys: {missing}, {extra}')
        if a.mode == 'staged':
            model.encoder.blocks.requires_grad_(False)
    else:
        model = make_model(a.mode, state)
    if a.phase_head_only:
        model.requires_grad_(False)
        model.phase_head.requires_grad_(True)
    model.cuda()
    order_head = None
    if a.order_weight:
        with torch.random.fork_rng(devices=[]):
            order_head = nn.Linear(model.encoder.num_features, 1, bias=False).cuda()
            nn.init.zeros_(order_head.weight)
    teacher = None
    if a.distill:
        with torch.random.fork_rng(devices=[]):
            teacher = DomainVideoMAE().cuda().eval().requires_grad_(False)
        teacher.load_state_dict(state)
    del state
    num_params = sum(t.numel() for t in model.parameters())
    initial_trainable = sum(t.numel() for t in model.parameters() if t.requires_grad)
    config = {'mode': a.mode, 'seed': a.seed, 'batch': a.batch, 'epochs': a.epochs,
              'base_lr': a.base_lr, 'layer_decay': .85, 'head_multiplier': 5,
              'continue_dual': a.continue_dual, 'class_power': a.class_power, 'ir_jitter': a.ir_jitter,
              'keep_blocks_frozen': a.keep_blocks_frozen,
              'domain_variant': a.domain_variant, 'mask_rate': a.mask_rate, 'distill': a.distill,
              'initial_sha256': sha(a.initial), 'cache_sha256': sha(a.cache/'manifest.json'),
              'parameters': num_params, 'initial_trainable': initial_trainable,
              'train_samples': len(train), 'validation_samples': len(valid), 'test_accessed': False}
    use_alignment_loss = a.alignment not in ('none', 'control')
    if a.alignment != 'none':
        config['alignment'] = a.alignment
    if use_alignment_loss:
        config.update(alignment=a.alignment, alignment_weight=.04, alignment_tokens=32, alignment_temperature=.1)
    if order_head is not None:
        config.update(order_weight=a.order_weight, order_training_parameters=order_head.in_features)
    if a.phase_head_only:
        config['phase_head_only'] = True
    if a.pace_jitter:
        config.update(pace_jitter_max=.13, pace_jitter_probability=.5)
    write(a.output/'config.json', config)
    started = time.monotonic()
    if a.mode == 'ssl':
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        ssl_loss = adapt_without_labels(model, tr)
        write(a.output/'ssl.json', {'training_clips': len(train), 'loss': ssl_loss, 'labels_used': False})
        print(json.dumps({'ssl_loss': ssl_loss}), flush=True)
        # Keep the supervised random stream matched to the staged-only comparison.
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
    groups = model.layer_groups(a.base_lr, .85, 5, .05)
    # Latter four blocks are unfrozen after epoch one; retain their optimizer groups.
    unfreeze_late = a.mode in ('staged', 'ssl') and not a.keep_blocks_frozen
    delayed = [q for block in model.encoder.blocks[-4:] for q in block.parameters()] if unfreeze_late else []
    eligible = {id(t) for t in model.parameters() if t.requires_grad} | {id(t) for t in delayed}
    phase_parameters = list(model.phase_head.parameters()) if hasattr(model, 'phase_head') else []
    eligible.difference_update(id(t) for t in phase_parameters)
    patch_ids = {id(t) for t in model.encoder.patch_embed.parameters()}
    patch_ids.update(id(t) for name, t in model.named_parameters() if 'temporal_adapter' in name)
    optimizer_groups = []
    for group in groups:
        chosen = [q for q in group['params'] if id(q) in eligible]
        for is_patch in (False, True):
            selected = [q for q in chosen if (id(q) in patch_ids) == is_patch]
            if selected:
                scale = 2.5 if is_patch and a.mode != 'control' else group['lr_scale']
                optimizer_groups.append(dict(group, params=selected, lr_scale=scale))
    if phase_parameters:
        optimizer_groups.append({'params': phase_parameters, 'lr_scale': 5., 'weight_decay': .05})
    if order_head is not None:
        optimizer_groups.append({'params': list(order_head.parameters()), 'lr_scale': 10., 'weight_decay': .05})
    optimizer = torch.optim.AdamW(optimizer_groups)
    gradient_parameters = list(model.parameters()) + (list(order_head.parameters()) if order_head is not None else [])
    ema = ModelEmaV3(model, decay=.99, device='cuda')
    counts = np.bincount([r['label'] for r in train.rows], minlength=40)
    if not np.all(counts > 0):
        raise ValueError('A class is absent from training')
    weights = counts.astype(np.float32) ** -a.class_power
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights / weights.mean(), device='cuda'), label_smoothing=.05)
    best, best_epoch, step = (-1., -1.), 0, 0
    ir_rng = torch.Generator(device='cuda').manual_seed(a.seed + 397)
    mask_rng = torch.Generator(device='cuda').manual_seed(a.seed + 871)
    align_rng = torch.Generator(device='cuda').manual_seed(a.seed + 1259)
    pace_rng = torch.Generator(device='cuda').manual_seed(a.seed + 1597)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, a.epochs + 1):
        if epoch == 2:
            for q in delayed:
                q.requires_grad_(True)
        model.train(not a.phase_head_only)
        seen, loss_sum = 0, 0.
        alignment_sum = torch.zeros((), device='cuda')
        order_sum = torch.zeros((), device='cuda')
        for x, y, _ in tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if a.pace_jitter:
                # Move the interior positions while retaining endpoints and frame order.
                frame_time = torch.linspace(0., 1., x.size(2), device=x.device)
                draws = torch.rand((len(y), 2), device=x.device, generator=pace_rng)
                strength = (draws[:, :1] * .26 - .13) * (draws[:, 1:] < .5)
                warped = frame_time + strength * torch.sin(math.pi * frame_time)
                indices = (warped * (x.size(2) - 1)).round().long().clamp(0, x.size(2) - 1)
                x = x.gather(2, indices[:, None, :, None, None].expand_as(x))
            if a.ir_jitter:
                # One gain and offset per clip, with depth pixels and alignment preserved.
                jitter = torch.rand((len(y), 2, 1, 1, 1), generator=ir_rng, device=x.device)
                gain, offset = .75 + .5*jitter[:, :1], -.05 + .1*jitter[:, 1:]
                x[:, 3:4] = (x[:, 3:4] * gain + offset).clamp(0., 1.)
            frac = step / max(1, a.epochs * len(tr) - 1)
            schedule = min(1., .2 + 4. * frac) * (.15 + .85 * .5 * (1 + math.cos(math.pi * frac)))
            for group in optimizer.param_groups:
                group['lr'] = a.base_lr * group['lr_scale'] * schedule
            optimizer.zero_grad(set_to_none=True)
            perm = torch.randperm(len(y), device='cuda') if random.random() < .5 else None
            mix = float(np.random.beta(.2, .2)) if perm is not None else 1.
            if perm is not None:
                x = mix * x + (1. - mix) * x[perm]
            if a.mask_rate:
                clean = x
                batch, _, frames, height, width = x.shape
                patches = torch.rand((batch, 1, 8, 14, 14), generator=mask_rng, device=x.device) < a.mask_rate
                patches = F.interpolate(patches.float(), (frames, height, width), mode='nearest').bool()
                mask_ir = torch.rand((batch, 1, 1, 1, 1), generator=mask_rng, device=x.device) < .5
                mask = torch.cat((~mask_ir.expand(-1, 3, -1, -1, -1), mask_ir), dim=1) & patches
                x = torch.where(mask, model.mean, x)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if a.alignment != 'none':
                    logits, depth, ir = model.forward_with_streams(x)
                elif order_head is not None:
                    logits, order_tokens = model.forward_with_order_inputs(x)
                else:
                    logits = model(x)
                loss = criterion(logits, y)
                if perm is not None:
                    loss = mix * loss + (1. - mix) * criterion(logits, y[perm])
                if teacher is not None:
                    with torch.no_grad():
                        target = (teacher(clean).float() / 2).softmax(dim=1)
                    kl = F.kl_div((logits.float() / 2).log_softmax(dim=1), target, reduction='batchmean')
                    loss = loss + a.distill * 4 * kl
                if use_alignment_loss:
                    aux = local_alignment(x, depth, ir, a.alignment, align_rng)
                    loss = loss + .04 * aux
                    alignment_sum += aux.detach() * len(y)
                if order_head is not None:
                    phases = order_tokens.reshape(len(y), 8, -1, order_tokens.size(-1)).mean(dim=2)
                    phases = F.layer_norm(phases.float(), (phases.size(-1),))
                    direction = phases[:, 2:] - phases[:, :-2]
                    # Reversing each pair negates its score, so the two labels have equal loss.
                    order_loss = F.softplus(-order_head(direction).float()).mean()
                    loss = loss + a.order_weight * order_loss
                    order_sum += order_loss.detach() * len(y)
            loss.backward()
            nn.utils.clip_grad_norm_(gradient_parameters, 5., error_if_nonfinite=True)
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
        if use_alignment_loss:
            row['alignment_loss'] = float(alignment_sum) / seen
        if order_head is not None:
            row['order_loss'] = float(order_sum) / seen
        with (a.output/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
    write(a.output/'summary.json', dict(mode=a.mode, best_epoch=best_epoch, accuracy=best[0], macro_f1=best[1]))


if __name__ == '__main__':
    main()
