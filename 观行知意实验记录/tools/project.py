"""项目操作入口。所有路径相对本项目，建议先读 docs/04_代码导读.md。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'baseline'
sys.path.insert(0, str(BASE))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(*args):
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, env=env)


def check_data(args):
    # 复用官方校验算法：逐帧检查图像，并将文件名与内容组合为片段SHA256。
    from evaluation_adapters.depth_ir_dataset import clip_digest, paired_files
    from training.prepare_training_manifest import combine
    raw = ROOT / 'data/raw'
    output = ROOT / 'runs/data-check'
    output.mkdir(parents=True, exist_ok=True)
    combined = output / 'training-manifest.json'
    if combined.exists():
        # 官方combine以独占方式创建；仅清理本工具自己生成的这一文件。
        combined.unlink()
    combine(raw, combined)
    merged = read(combined)
    jobs, counts, classes = [], {}, merged['classes']
    for split, expected in [('train', 1995), ('validation', 466)]:
        manifest = read(raw / split / 'manifest.json')
        if manifest['classes'] != classes or len(manifest['samples']) != expected:
            raise ValueError(f'{split}: 数量或类别顺序不符')
        counts[split] = [0] * 40
        for sample in manifest['samples']:
            if type(sample['label']) is not int or not 0 <= sample['label'] < 40:
                raise ValueError('标签必须是0到39的整数')
            counts[split][sample['label']] += 1
            jobs.append((split, sample))

    def inspect(job):
        split, sample = job
        p = (raw / split / sample['file']).resolve()
        if not p.is_relative_to((raw / split).resolve()):
            raise ValueError('清单路径越界')
        digest = clip_digest(p, verify_images=True)
        if digest != sample['sha256']:
            raise ValueError(f'内容校验不匹配: {p}')
        paired = paired_files(p)
        n = len(paired['Depth_Color'])
        if not 1 <= n <= 32:
            raise ValueError(f'帧数不符: {p}')
        return n

    frames, start = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, n in enumerate(pool.map(inspect, jobs), 1):
            frames.append(n)
            if i % 100 == 0:
                print(f'已校验 {i}/{len(jobs)} 段', flush=True)
    chinese = '洗脸,刷牙,梳头,脱衣服,擦手,穿衣服,喝水,吃东西,拿取并使用餐具,倒饮料,搅拌饮料,削水果,扫地,拖地,擦碗,擦窗户和桌子,叠衣服,敲键盘,写字,打电话,看时间,阅读文档,翻页,戴耳机听音乐,使用手机,看电视,玩游戏,自拍,原地慢跑,深蹲,开合跳,伸展运动,站起,躺下,坐下,弓步,走路,吃药,自我按摩,测量体温'.split(',')
    with (ROOT / 'data/classes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_id', 'class_name', '中文解释', 'train_count', 'validation_count'])
        for i, name in enumerate(classes):
            w.writerow([i, name, chinese[i], counts['train'][i], counts['validation'][i]])
    result = {'passed': True, 'samples': len(jobs), 'counts': counts,
              'all_images_decoded_and_sha256_verified': True,
              'frame_range': [min(frames), max(frames)],
              'paired_images': 2 * sum(frames),
              'subjects': {s: sorted({x['subject'] for x in merged['samples'] if x['split'] == s}) for s in counts},
              'seconds': time.time() - start,
              'manifest_sha256': {s: sha(raw / s / 'manifest.json') for s in counts}}
    write(output / 'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'counts'}, ensure_ascii=False), flush=True)


def preview(args):
    from PIL import Image, ImageDraw
    from depth_ir_io import sample_frame_lists, uniform_indices
    raw = ROOT / 'data/raw'
    m = read(raw / args.split / 'manifest.json')
    if not 0 <= args.index < len(m['samples']):
        raise ValueError('样本序号超出范围')
    row = m['samples'][args.index]
    depth, ir = sample_frame_lists(raw / args.split / row['file'])
    panel = Image.new('RGB', (6 * 228, 2 * 194 + 45), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{args.split} | {row['file']} | {m['classes'][row['label']]}", fill='black')
    for col, idx in enumerate(uniform_indices(len(depth), 6)):
        for y, paths in enumerate([depth, ir]):
            with Image.open(paths[idx]) as im:
                panel.paste(im.convert('RGB').resize((228, 171)), (col * 228, 45 + y * 194))
            draw.text((col * 228 + 4, 45 + y * 194 + 174), f"{'Depth' if y == 0 else 'IR'} frame {idx}", fill='black')
    path = ROOT / 'runs/preview.png'
    panel.save(path)
    print(path)


def prepare(args):
    from training.prepare_training_manifest import combine
    manifest = ROOT / 'cache/training-manifest.json'
    if not manifest.exists():
        combine(ROOT / 'data/raw', manifest)
    else:
        # 拒绝沿用数据内容已变化的旧清单，避免缓存与原始数据脱节。
        rows = read(manifest)['samples']
        for split in ['train', 'validation']:
            current = sha(ROOT / 'data/raw' / split / 'manifest.json')
            if any(x['published_manifest_sha256'] != current for x in rows if x['split'] == split):
                raise ValueError('原始清单已变化，请另建缓存并重新生成清单')
    target = ROOT / ('cache/small16' if args.small else 'cache/full16')
    if (target / 'manifest.json').exists():
        meta = read(target / 'manifest.json')
        if meta['shape_per_sample'] != [4, 16, 128, 128] or bool(meta['smoke']) != args.small:
            raise ValueError('已有缓存规格不匹配')
        print(f'已有完整缓存，复用: {target}')
        return
    source = manifest
    if args.small:
        data = read(manifest)
        # 各取四个不同动作，训练与验证仍严格保留原始划分。
        data['samples'] = [next(x for x in data['samples'] if x['split'] == s and x['label'] == c)
                           for s in ['train', 'validation'] for c in [0, 6, 12, 28]]
        source = ROOT / 'cache/small-manifest.json'
        write(source, data)
    command = [BASE / 'training/build_cache.py', '--manifest', source,
               '--output', target, '--frames', '16', '--size', '128', '--include-ir']
    if args.small:
        command.append('--smoke')
    run(*command)


def environment(args):
    import numpy as np
    import PIL
    import torch
    import torchvision
    result = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'numpy': np.__version__, 'Pillow': PIL.__version__,
              'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到CUDA；本机方案要求GPU实际运算成功')
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError('CUDA梯度异常')
    result.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                  architectures=torch.cuda.get_arch_list(), cuda_forward_backward_passed=True)
    write(ROOT / 'runs/environment-check.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def smoke(args):
    import numpy as np
    import torch
    from raw_model import RawModel
    from training.train_raw import CachedClips
    from download_pretrained import SHA
    if args.steps < 1:
        raise ValueError('steps必须为正整数')
    weights = ROOT / 'weights/pretrained.pth'
    if sha(weights) != SHA:
        raise ValueError('公开预训练权重SHA256不匹配')
    output = ROOT / 'runs/smoke'
    if (output / 'smoke.pt').exists():
        raise FileExistsError('已有smoke结果，请先保留或重命名runs/smoke后再运行')
    output.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('本机验证需要CUDA')
    model = RawModel(channels=4, pretrained=weights).cuda().train()
    dataset = CachedClips(ROOT / 'cache/small16', 'train', training=True)
    # batch=1，workers=0；直接取Dataset样本便于新手调试。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # 默认65536在这组预训练参数的首次FP16反向中会溢出。
    # 小规模验证采用保守初始缩放；仍检查反缩放后的梯度确实有限。
    scaler = torch.amp.GradScaler('cuda', init_scale=128)
    before = model.head[1].weight.detach().clone()
    losses, norms = [], []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        x, label, _ = dataset[step % len(dataset)]
        x = x.unsqueeze(0).cuda()  # [4,16,128,128] -> [1,4,16,128,128]
        y = torch.tensor([label], device='cuda', dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)  # [1,40]，这里是分数，不是类别编号。
            loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError('损失出现NaN或Inf')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError('梯度无效')
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        norms.append(norm.item())
        print(f'step={step + 1}, loss={loss.item():.6f}, grad_norm={norm.item():.6f}', flush=True)
    changed = not torch.equal(before, model.head[1].weight.detach())
    if not changed:
        raise RuntimeError('分类头参数没有实际更新')
    # 保存普通state_dict，组织方无需训练缓存、优化器或预训练文件即可加载。
    torch.save(model.cpu().state_dict(), output / 'smoke.pt')
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 100_000_000:
        raise RuntimeError('参数超过100M')
    config = read(BASE / 'config.json')
    config.update(weights_file='smoke.pt', person_crop=False)
    write(output / 'inference-config.json', config)
    metadata = {'smoke': True, 'formal_submission_ready': False, 'steps': args.steps,
                'seed': 42, 'batch': 1, 'workers': 0, 'losses': losses, 'gradient_norms': norms,
                'head_parameters_changed': changed, 'parameters': parameters,
                'pretrained_sha256': SHA, 'checkpoint_sha256': sha(output / 'smoke.pt'),
                'cache_manifest_sha256': sha(ROOT / 'cache/small16/manifest.json'),
                'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20,
                'seconds': time.time() - started, 'torch': torch.__version__,
                'amp_dtype': 'float16', 'grad_scaler_initial_scale': 128,
                'note': '仅用于工程流程验证；不代表识别效果，不可作为最终比赛权重'}
    write(output / 'metadata.json', metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def train(args):
    config = read(ROOT / 'environment/train-config.json')
    # 命令行覆盖只影响本次实验，保留默认30轮配置。
    for key in ['epochs', 'batch', 'workers', 'lr']:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if config['epochs'] < 1 or config['batch'] < 1 or config['workers'] < 0 or config['lr'] <= 0:
        raise ValueError('epochs/batch/lr必须为正数，workers不得为负数')
    output = ROOT / 'runs' / args.name
    if not (ROOT / 'cache/full16/manifest.json').exists():
        raise FileNotFoundError('先运行 prepare 构建完整缓存')
    command = [BASE / 'training/train_raw.py', '--cache', ROOT / 'cache/full16',
               '--pretrained', ROOT / 'weights/pretrained.pth', '--output', output]
    for key in ['batch', 'epochs', 'workers', 'lr', 'seed', 'device']:
        command += ['--' + key, str(config[key])]
    run(*command)
    write(output / 'train-config.json', config)
    inference = read(BASE / 'config.json')
    inference.update(weights_file='best_ema.pt', person_crop=False)
    write(output / 'inference-config.json', inference)
    write(output / 'selection.json', {'weights_file': 'best_ema.pt',
          'selection': '公开验证集准确率最高的EMA检查点；相同准确率保留较早轮次'})


def predict(args):
    import numpy as np
    import torch
    from inference import Model
    from analyze_predictions import analyze, render
    output = ROOT / args.run
    config = read(output / 'inference-config.json')
    config['weights_path'] = str(output / config['weights_file'])
    config['package_dir'] = str(BASE)
    metadata = read(output / 'metadata.json')
    demo = metadata.get('smoke', False)
    if not demo and args.limit:
        raise ValueError('正式模型预测必须覆盖整个验证集；小样本检查请使用runs/smoke')
    manifest = read(ROOT / 'data/raw/validation/manifest.json')
    if args.limit:
        # 工程演示取四个不同动作，而不是只取排在最前面的同类片段。
        samples = read(ROOT / 'cache/small-manifest.json')['samples']
        ids = {x['sample_id'] for x in samples if x['split'] == 'validation'}
        manifest['samples'] = [x for x in manifest['samples'] if x['file'] + '.npy' in ids][:args.limit]
    model = Model(config, 'cuda' if torch.cuda.is_available() else 'cpu')
    rows, latency = [], []
    for i, sample in enumerate(manifest['samples']):
        path = ROOT / 'data/raw/validation' / sample['file']
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        scores = np.asarray(model.predict(str(path)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i:
            latency.append((time.perf_counter() - start) * 1000)
        if scores.shape != (40,) or not np.isfinite(scores).all():
            raise ValueError('predict必须返回40个有限分数')
        rows.append([sample.get('sample_id', sample['file'] + '.npy'), int(scores.argmax())])
        if (i + 1) % 50 == 0:
            print(f'预测 {i + 1}/{len(manifest["samples"])}', flush=True)
    name = 'validation-smoke' if demo else 'validation'
    csv_path = output / f'{name}.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'predicted_class'])
        w.writerows(rows)
    manifest_path = output / f'{name}-manifest.json'
    write(manifest_path, manifest)
    report = analyze(csv_path, manifest_path)
    report.update(smoke=demo, formal_submission_ready=not demo and len(rows) == 466,
                  local_pipeline_latency_ms_mean=float(np.mean(latency)) if latency else None,
                  timing_note='本地端到端耗时；排除首样本；不是组织方正式计时结果')
    write(output / f'{name}-report.json', report)
    (output / f'{name}-report.html').write_text(render(report), encoding='utf-8')
    print(json.dumps({'samples': len(rows), 'smoke': demo, 'metrics': report['metrics'],
                      'latency_ms_mean': report['local_pipeline_latency_ms_mean']}, ensure_ascii=False))


def package(args):
    from analyze_predictions import analyze
    output = ROOT / args.run
    config = read(output / 'inference-config.json')
    metadata = read(output / 'metadata.json')
    smoke_run = metadata.get('smoke', False)
    if smoke_run and not args.demo:
        raise ValueError('smoke权重禁止打包成正式作品；工程检查须加--demo')
    if not args.demo:
        with (output / 'validation.csv').open(encoding='utf-8-sig', newline='') as f:
            if next(csv.reader(f), None) != ['sample_id', 'predicted_class']:
                raise ValueError('CSV列名和顺序必须严格为sample_id,predicted_class')
        result = analyze(output / 'validation.csv', ROOT / 'data/raw/validation/manifest.json')
        if result['samples'] != 466:
            raise ValueError('正式CSV必须覆盖466段')
    dest = ROOT / 'submission' / ('demo' if args.demo else args.name)
    dest.mkdir(parents=True, exist_ok=True)
    weights = output / config['weights_file']
    shutil.copy2(weights, dest / 'model.pt')
    config['weights_file'] = 'model.pt'
    config.pop('weights_path', None)
    config.pop('package_dir', None)
    files = ['inference.py', 'raw_model.py', 'ig65m_backbone.py', 'depth_ir_io.py',
             'build_cache.py', 'predict.py', 'analyze_predictions.py', 'download_pretrained.py',
             'requirements.txt', 'training/train_raw.py', 'training/raw_model.py',
             'training/ig65m_backbone.py', 'training/depth_ir_io.py', 'training/build_cache.py',
             'training/prepare_training_manifest.py']
    path = dest / ('demo-code.zip' if args.demo else 'code.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(BASE / name, name)
        z.writestr('config.json', json.dumps(config, indent=2))
        z.writestr('README.md', (ROOT / 'docs/08_提交包运行说明.md').read_text(encoding='utf-8'))
        z.writestr('environment-reference.json', (ROOT / 'environment/official-reference.json').read_text(encoding='utf-8'))
        z.writestr('classes.json', json.dumps(read(ROOT / 'data/raw/train/manifest.json')['classes']))
        license_path = ROOT / 'competition/ig65m-LICENSE.txt'
        if license_path.is_file():
            z.write(license_path, 'ig65m-LICENSE.txt')
    with zipfile.ZipFile(path) as z:
        count, unpacked = len(z.infolist()), sum(i.file_size for i in z.infolist())
        if count > 256 or unpacked > 64 * 1024**2 or path.stat().st_size > 20 * 1024**2:
            raise ValueError('代码包超过赛事体积或文件数限制')
    sample = read(ROOT / 'data/raw/validation/manifest.json')['samples'][0]['file']
    run(ROOT / 'tools/offline_probe.py', '--package', path, '--weights', dest / 'model.pt',
        '--sample', ROOT / 'data/raw/validation' / sample, '--output', dest / 'offline-check.json')
    info = {'demo': args.demo, 'formal_submission_ready': False,
            'code_and_weights_checked': True,
            'remaining_manual_materials': ['技术报告PDF', '网页说明', '团队信息和QQ权重发送', 'Linux参考环境复现'],
            'code_zip_bytes': path.stat().st_size, 'files': count, 'unpacked_bytes': unpacked,
            'code_zip_sha256': sha(path), 'weights_sha256': sha(dest / 'model.pt'),
            'weights_file': 'model.pt', 'linux_verified': False}
    write(dest / 'package-check.json', info)
    if not args.demo:
        shutil.copy2(output / 'validation.csv', dest / 'validation.csv')
    print(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('check-data', help='逐帧解码、内容哈希、同步关系与人物划分检查')
    p.add_argument('--workers', type=int, default=8)
    p.set_defaults(func=check_data)
    p = commands.add_parser('preview', help='生成同步深度图/IR六帧预览')
    p.add_argument('--split', choices=['train', 'validation'], default='train')
    p.add_argument('--index', type=int, default=0)
    p.set_defaults(func=preview)
    p = commands.add_parser('prepare', help='生成官方清单和缓存；完整缓存约2.4GiB')
    p.add_argument('--small', action='store_true')
    p.set_defaults(func=prepare)
    p = commands.add_parser('check-env', help='实际GPU前向和反向检查')
    p.set_defaults(func=environment)
    p = commands.add_parser('smoke', help='少量真实训练步骤，不能作为正式模型')
    p.add_argument('--steps', type=int, default=3)
    p.set_defaults(func=smoke)
    p = commands.add_parser('train', help='正式完整基线训练，耗时较长')
    p.add_argument('--name', default='full16')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch', type=int)
    p.add_argument('--workers', type=int)
    p.add_argument('--lr', type=float)
    p.set_defaults(func=train)
    p = commands.add_parser('predict', help='使用已保存权重预测并计算40类指标')
    p.add_argument('--run', default='runs/full16')
    p.add_argument('--limit', type=int, default=0)
    p.set_defaults(func=predict)
    p = commands.add_parser('package', help='打包代码，单独导出权重并离线验证')
    p.add_argument('--run', default='runs/full16')
    p.add_argument('--name', default='final')
    p.add_argument('--demo', action='store_true')
    p.set_defaults(func=package)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
