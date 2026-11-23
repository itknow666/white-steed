"""Build matched video caches, decoding each sampled frame once per clip."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'baseline'))
from depth_ir_io import frame_paths, resize_frame, synchronized_paths, uniform_indices


def render_paths(depth, infrared, crop, specs):
    synchronized_paths(depth, infrared)
    indices = {frames: uniform_indices(len(depth), frames) for frames in {key[0] for key in specs}}
    arrays = {key: np.empty((4, key[0], key[1], key[1]), dtype=np.uint8) for key in specs}
    unique_indices = set().union(*(set(sequence) for sequence in indices.values()))
    for mode, paths, channel in (('RGB', depth, slice(0, 3)), ('L', infrared, slice(3, 4))):
        for idx in unique_indices:
            with Image.open(paths[idx]) as source:
                if source.mode.startswith('I') or source.mode == 'F':
                    raise ValueError('Expected published 8-bit frames')
                frame = source.convert(mode)
            resized = {}
            for key, array in arrays.items():
                frames, size, geometry = key
                slots = np.flatnonzero(indices[frames] == idx)
                if len(slots):
                    if (size, geometry) not in resized:
                        pixels = np.asarray(resize_frame(frame, size, crop, geometry))
                        resized[size, geometry] = pixels.transpose(2, 0, 1) if mode == 'RGB' else pixels[None]
                    array[channel, slots] = resized[size, geometry][:, None]
    return arrays


def render(row, frames, variants):
    depth = frame_paths(row['depth_dir'])
    infrared = frame_paths(row['ir_dir'])
    clips = render_paths(depth, infrared, row['crop'], [(frames, *key) for key in variants])
    return {key[1:]: value for key, value in clips.items()}


def build(source, out, frames, workers):
    started = perf_counter()
    source_bytes = source.read_bytes()
    meta = json.loads(source_bytes)
    counts = meta['counts']
    rows = meta['samples']
    if counts != {'train':1995, 'validation':466} or len(rows) != 2461 or meta.get('smoke'):
        raise ValueError('A complete official cache is required')
    variants = [(size, geometry) for size in (192, 224) for geometry in ('stretch', 'letterbox')
                if frames != 16 or geometry == 'letterbox']
    folders, arrays = {}, {}
    for size, geometry in variants:
        folder = out/f'f{frames}-s{size}-{geometry}'
        folder.mkdir(parents=True)
        folders[size, geometry] = folder
        arrays[size, geometry] = {
            split: np.lib.format.open_memmap(folder/(split+'.npy'), mode='w+', dtype=np.uint8,
                                            shape=(count,4,frames,size,size))
            for split, count in counts.items()
        }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(rows), 24):
            batch = rows[start:start+24]
            for row, rendered in zip(batch, executor.map(lambda row: render(row, frames, variants), batch)):
                for key, arr in rendered.items():
                    arrays[key][row['split']][row['row']] = arr
            if start % 240 == 0 or start + len(batch) == len(rows):
                print(f'{frames} frames: {start+len(batch)}/{len(rows)} clips in {perf_counter()-started:.1f}s', flush=True)
    for (size, geometry), folder in folders.items():
        for array in arrays[size, geometry].values():
            array.flush()
        output = dict(meta, shape_per_sample=[4,frames,size,size], geometry=geometry,
                      crop_reuse_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
                      cache_seconds=perf_counter()-started)
        temp = folder/'manifest.partial.json'
        temp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(folder/'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', required=True, type=int, choices=(16,24,32))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    build(args.source, args.output, args.frames, args.workers)
