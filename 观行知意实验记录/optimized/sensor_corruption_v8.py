"""Synchronized, clip-consistent synthetic perturbations of depth and IR."""
import torch


def corrupt_training_clip(x, mean, rng, kind='mixed'):
    """Darken IR or hide a depth region, using one choice for the whole clip."""
    draws = torch.rand((len(x), 6), device=x.device, generator=rng)
    active = draws[:, 0] < .55
    dark_ir = active & (draws[:, 1] < .5) & (kind == 'mixed')
    depth_gap = active & ~dark_ir
    center_y, center_x = .25 + .5 * draws[:, 2], .25 + .5 * draws[:, 3]
    half = (.18 + .16 * draws[:, 4]) / 2
    height, width = x.shape[-2:]
    rows = (torch.arange(height, device=x.device) + .5) / height
    cols = (torch.arange(width, device=x.device) + .5) / width
    within_rows = (rows[None, :] - center_y[:, None]).abs() < half[:, None]
    within_cols = (cols[None, :] - center_x[:, None]).abs() < half[:, None]
    patch = (depth_gap[:, None, None] & within_rows[:, :, None] & within_cols[:, None, :])
    depth = torch.where(patch[:, None, None], mean[:, :3], x[:, :3])
    gain = torch.where(dark_ir, .55 + .35 * draws[:, 5], 1.)
    ir = x[:, 3:] * gain[:, None, None, None, None]
    return torch.cat((depth, ir), dim=1), active


def stress_clip(x, mean, case):
    if case == 'clean':
        return x
    altered = x.clone()
    if case == 'dark-ir':
        altered[:, 3:].mul_(.50)
    elif case == 'depth-gap':
        h, w = x.shape[-2:]
        altered[:, :3, :, 3*h//10:7*h//10, 3*w//10:7*w//10] = mean[:, :3]
    elif case == 'short-offset-gap':
        frames, h, w = x.shape[-3:]
        altered[:, :3, frames//4:3*frames//4, h//8:h//2, w//2:7*w//8] = 0.
    else:
        raise ValueError(f'Unknown synthetic stress case {case}')
    return altered
