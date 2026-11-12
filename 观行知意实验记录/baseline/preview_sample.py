"""Save eight synchronized depth/IR frame pairs from a published clip."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from depth_ir_io import frame_key, sample_frame_lists, uniform_indices


def resolve_sample(dataset, sample_id):
    dataset = Path(dataset).resolve()
    name = sample_id[:-4] if sample_id.endswith('.npy') else sample_id
    candidate = (dataset / name).resolve()
    if not candidate.is_relative_to(dataset):
        raise ValueError('sample_id must stay inside the dataset')
    if candidate.is_dir():
        return candidate
    manifest = dataset / 'manifest.json'
    if manifest.exists():
        for item in json.loads(manifest.read_text(encoding='utf-8'))['samples']:
            if sample_id in (item.get('sample_id'), item['file'], item['file'] + '.npy'):
                candidate = (dataset / item['file']).resolve()
                if candidate.is_relative_to(dataset) and candidate.is_dir():
                    return candidate
    raise ValueError(f'Sample not found: {sample_id}')


def render_preview(sample, output):
    sample, output = Path(sample), Path(output)
    depth, ir = sample_frame_lists(sample)
    if len(depth) != len(ir):
        raise ValueError('Depth and IR must have the same frame count')
    for left, right in zip(depth, ir):
        a, b = frame_key(left), frame_key(right)
        if a[0] == 0 and b[0] == 0 and a[1] != b[1]:
            raise ValueError('Depth and IR frame numbers do not match')
    # Preserve the published 4:3 frame shape. Four columns keep labels readable.
    width, height, gap, label = 304, 228, 12, 26
    font = ImageFont.load_default(size=17)
    canvas = Image.new('RGB', (4 * (width + gap) + gap, 2 * (2 * (height + label) + gap) + 68), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), f'Sample: {sample.name}', fill='black', font=font)
    draw.text((gap, 36), f'Depth_Color / IR | {len(depth)} paired frames | 8 time points', fill='black', font=font)
    selected = []
    for slot, index in enumerate(uniform_indices(len(depth), 8)):
        x = gap + (slot % 4) * (width + gap)
        y = 68 + (slot // 4) * (2 * (height + label) + gap)
        for row, (modality, path) in enumerate((('Depth_Color', depth[index]), ('IR', ir[index]))):
            top = y + row * (height + label)
            number = frame_key(path)[1]
            draw.text((x, top), f'{modality} | frame {number} ({index + 1}/{len(depth)})', fill='black', font=font)
            with Image.open(path) as original:
                image = ImageOps.contain(original.convert('RGB'), (width, height), Image.Resampling.BILINEAR)
                canvas.paste(image, (x + (width - image.width) // 2, top + label + (height - image.height) // 2))
        selected.append({'index': int(index), 'depth': depth[index].name, 'ir': ir[index].name})
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {'sample_id': sample.name + '.npy', 'paired_frames': len(depth), 'selected': selected,
            'output': str(output), 'image_size': list(canvas.size)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sample-id', required=True, help='sample_id from predictions.csv or the error table')
    parser.add_argument('--output', type=Path, default=Path('sample-preview.png'))
    args = parser.parse_args()
    print(json.dumps(render_preview(resolve_sample(args.dataset, args.sample_id), args.output), indent=2))
