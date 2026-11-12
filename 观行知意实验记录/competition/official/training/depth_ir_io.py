"""Shared image loading for training and Model.predict. Arrays use C,T,H,W."""
from pathlib import Path
import json
import re
import numpy as np
from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def natural_key(path):
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                 for part in re.split(r'(\d+)', str(path)))


def frame_key(path):
    match = re.search(r'_(\d+)(?:_Color)?\.[^.]+$', path.name, re.IGNORECASE)
    return (0, int(match.group(1)), natural_key(path.name)) if match else (1, 0, natural_key(path.name))


def frame_paths(directory):
    directory = Path(directory)
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=frame_key)
    if not paths:
        raise ValueError(f'No image frames in {directory}')
    if any(p.is_symlink() for p in paths):
        raise ValueError('Image frames must be regular files')
    return paths


def uniform_indices(length, frames):
    if length < 1 or frames < 1:
        raise ValueError('Frame counts must be positive')
    return np.linspace(0, length - 1, frames).round().astype(np.int64)


def _read(paths, frames, size, mode, crop):
    result = []
    for index in uniform_indices(len(paths), frames):
        with Image.open(paths[index]) as source:
            if source.mode.startswith('I') or source.mode == 'F':
                raise ValueError('This loader expects published 8-bit Depth_Color/IR; specify scaling for numeric depth')
            image = source.convert(mode)
            if crop is not None:
                x0, y0, x1, y1 = map(float, crop)
                if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
                    raise ValueError('crop must be normalized x0,y0,x1,y1')
                w, h = image.size
                box = (round(x0*w), round(y0*h), round(x1*w), round(y1*h))
                image = image.crop(box)
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            value = np.asarray(image, dtype=np.float32) / 255.0
            if value.ndim == 2:
                value = value[..., None]
            result.append(value.transpose(2, 0, 1))
    return np.stack(result, axis=1)


def read_frame_lists(depth_paths, ir_paths=None, *, frames=16, size=128, crop=None, depth_mode='color', include_ir=True):
    if size < 1 or depth_mode not in ('color', 'gray'):
        raise ValueError('size must be positive; depth_mode is color or gray')
    depth = _read(depth_paths, frames, size, 'RGB' if depth_mode == 'color' else 'L', crop)
    if depth_mode == 'gray':
        depth = np.repeat(depth, 3, axis=0)
    if include_ir:
        if not ir_paths:
            raise ValueError('IR frames are required for depth+IR input')
        ir = _read(ir_paths, frames, size, 'L', crop)
        depth = np.concatenate([depth, ir], axis=0)
    return np.ascontiguousarray(depth, dtype=np.float32)


def read_directories(depth_dir, ir_dir=None, **kwargs):
    ir_paths = frame_paths(ir_dir) if kwargs.get('include_ir', True) else None
    return read_frame_lists(frame_paths(depth_dir), ir_paths, **kwargs)


def sample_frame_lists(sample_path):
    root = Path(sample_path).resolve()
    manifest = root / 'frames.json'
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if set(data) != {'depth', 'ir'}:
            raise ValueError('frames.json needs depth and ir lists')
        lists = []
        for key in ('depth', 'ir'):
            if not isinstance(data[key], list) or not data[key]:
                raise ValueError(f'Empty {key} frame list')
            paths = []
            for name in data[key]:
                p = root / name
                if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
                    raise ValueError('Frame path escapes sample or is missing')
                paths.append(p)
            lists.append(paths)
        return tuple(lists)
    return frame_paths(root / 'Depth_Color'), frame_paths(root / 'IR')


def read_modalities(sample_path, **kwargs):
    depth, ir = sample_frame_lists(sample_path)
    return read_frame_lists(depth, ir, **kwargs)
