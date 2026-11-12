"""独立重新加载优化模型，评估实际推理入口并导出比赛代码包。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'baseline'))
from analyze_predictions import analyze, render


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--crop', action='store_true')
    p.add_argument('--package-name', required=True)
    p.add_argument('--flip', action='store_true')
    p.add_argument('--check-cache', type=Path)
    p.add_argument('--input-frames', type=int, help='显式的推理帧数对照，不改变训练权重')
    a = p.parse_args()
    run = a.run.resolve()
    runtime_name = f'runtime-control-{a.input_frames}-{"flip" if a.flip else "single"}' if a.input_frames is not None else 'runtime'
    runtime = run/runtime_name
    runtime.mkdir(exist_ok=True)
    for name in ['raw_model.py', 'ig65m_backbone.py', 'depth_ir_io.py', 'build_cache.py',
                 'predict.py', 'analyze_predictions.py', 'download_pretrained.py']:
        shutil.copy2(ROOT/'baseline'/name, runtime/name)
    shutil.copy2(ROOT/'optimized/inference.py', runtime/'inference.py')
    shutil.copy2(ROOT/'optimized/crop_policy.py', runtime/'crop_policy.py')
    if a.crop:
        shutil.copy2(ROOT/'baseline/yolo11n.pt', runtime/'yolo11n.pt')
    config = {'num_classes': 40, 'input_version': 2, 'channels': 4, 'depth_mode': 'color',
              'frames': 16, 'size': 128, 'weights_file': '../best_ema.pt', 'weights_format': 'pytorch',
              'person_crop': a.crop, 'flip_tta': a.flip, 'amp': True, 'channels_last_3d': True}
    if (run/'config.json').is_file():
        trained = json.loads((run/'config.json').read_text(encoding='utf-8'))
        cache_meta = json.loads((Path(trained['cache'])/'manifest.json').read_text(encoding='utf-8'))
        channels, frames, height, width = cache_meta['shape_per_sample']
        if channels != 4 or height != width or bool(cache_meta['person_crop']) != a.crop:
            raise ValueError('Inference/cache modality, shape or crop mismatch')
        config.update(frames=frames, size=height)
        config['crop_scale'] = cache_meta.get('crop_scale',1.4)
        config['frames'] = trained.get('input_frames', frames)
        config['architecture'] = trained.get('architecture', 'r34')
        config['ir_contrast'] = trained.get('ir_contrast',False)
        if config['ir_contrast']:
            shutil.copy2(ROOT/'optimized/ir_contrast.py',runtime/'ir_contrast.py')
        if config['architecture'] in ['swin3dt','ensemble']:
            shutil.copy2(ROOT/'optimized/video_swin.py', runtime/'video_swin.py')
        if config['architecture'] == 'ensemble':
            config['components'] = trained['components']
    if a.input_frames is not None:
        if a.input_frames not in [16, 32]:
            raise ValueError('This comparison supports 16 or 32 frames')
        config['frames'] = a.input_frames
    if a.crop:
        config['detector_file'] = 'yolo11n.pt'
    write(runtime/'config.json', config)
    sys.path.insert(0, str(runtime))
    # 该进程只评估一种配置，避免多模块加载时名字冲突。
    from inference import Model
    config.update(weights_path=str(run/'best_ema.pt'), package_dir=str(runtime))
    model = Model(config, 'cuda')
    manifest_path = ROOT/'data/raw/validation/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if a.check_cache:
        cache_meta = json.loads((a.check_cache/'manifest.json').read_text(encoding='utf-8'))
        array = np.load(a.check_cache/'validation.npy', mmap_mode='r')
        checks = []
        rows = [r for r in cache_meta['samples'] if r['split'] == 'validation']
        for row in [rows[0], rows[len(rows)//2], rows[-1]]:
            sample = ROOT/'data/raw/validation'/Path(row['depth_dir']).parent.name
            actual = model.prepare(sample)
            cached = array[row['row']].astype(np.float32)/255.
            error = float(np.max(np.abs(actual-cached)))
            if error > 1/255.+1e-6:
                raise ValueError(f'Cached/live preprocessing differs: {error}')
            checks.append({'sample_id': row['sample_id'], 'max_abs_error': error})
        check_name = f'preprocess-check-frames{a.input_frames}-control.json' if a.input_frames is not None else 'preprocess-check.json'
        write(run/check_name, {'passed': True, 'samples': checks})
    suffix = 'flip' if a.flip else 'single'
    if a.input_frames is not None:
        suffix += f'-frames{a.input_frames}-control'
    csv_path = run/f'validation-{suffix}.csv'
    elapsed, predictions = [], []
    for index, sample in enumerate(manifest['samples']):
        torch.cuda.synchronize()
        start = time.perf_counter()
        values = np.asarray(model.predict(ROOT/'data/raw/validation'/sample['file']))
        torch.cuda.synchronize()
        if index:
            elapsed.append((time.perf_counter()-start)*1000)
        if values.shape != (40,) or not np.isfinite(values).all():
            raise ValueError('Expected 40 finite scores')
        predictions.append([sample.get('sample_id', sample['file']+'.npy'), int(values.argmax())])
        if (index+1)%50 == 0:
            print(json.dumps({'run': a.run.name, 'mode': suffix, 'predicted': index+1, 'total': 466}), flush=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'predicted_class'])
        w.writerows(predictions)
    report = analyze(csv_path, manifest_path)
    report.update(flip_tta=a.flip, person_crop=a.crop, parameters=model.parameters,
                  local_pipeline_latency_ms_mean=float(np.mean(elapsed)),
                  weights_sha256=sha(run/'best_ema.pt'), linux_verified=False)
    write(run/f'validation-{suffix}-report.json', report)
    (run/f'validation-{suffix}-report.html').write_text(render(report), encoding='utf-8')
    dest = ROOT/'submission'/a.package_name
    dest.mkdir(exist_ok=True)
    config['weights_file'] = 'model.pt'
    config.pop('weights_path', None)
    config.pop('package_dir', None)
    archive = dest/'code.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in runtime.iterdir():
            if path.suffix in ['.py', '.pt']:
                z.write(path, path.name)
        z.writestr('config.json', json.dumps(config, indent=2))
        z.write(ROOT/'optimized/train.py', 'optimized/train.py')
        if (run/'config.json').is_file() and trained.get('training_source') in ['optimized/train_generalize.py','optimized/train_swin.py','optimized/train_ir.py','optimized/train_ablation.py']:
            training_source = trained['training_source']
            z.write(ROOT/training_source, training_source)
            if training_source == 'optimized/train_ablation.py':
                for helper in ['crop_policy.py','build_wide_cache.py','run_ablation_v4.py']:
                    z.write(ROOT/'optimized'/helper,'optimized/'+helper)
        if config.get('architecture') == 'ensemble':
            for name in ['train_generalize.py','train_swin.py','combine_models.py','compare_views.py',
                         'train_ablation.py','build_wide_cache.py','crop_policy.py','run_ablation_v4.py']:
                z.write(ROOT/'optimized'/name, 'optimized/'+name)
        for path in (ROOT/'baseline/training').glob('*.py'):
            z.write(path, 'training/'+path.name)
        z.writestr('requirements.txt', (ROOT/'environment/requirements-base.txt').read_text(encoding='utf-8')+
                   ('\nultralytics==8.3.228\nopencv-python-headless==4.13.0.92\n' if a.crop else '\n'))
        z.write(ROOT/'competition/ig65m-LICENSE.txt', 'ig65m-LICENSE.txt')
        if (ROOT/'competition/torchvision-LICENSE.txt').is_file():
            z.write(ROOT/'competition/torchvision-LICENSE.txt', 'torchvision-LICENSE.txt')
        if a.crop and (ROOT/'competition/ultralytics-LICENSE.txt').is_file():
            z.write(ROOT/'competition/ultralytics-LICENSE.txt', 'ultralytics-LICENSE.txt')
        z.writestr('classes.json', json.dumps(manifest['classes']))
        z.writestr('README.md', f"深度+IR四通道{config.get('architecture','r34')}，{config['frames']}帧{config['size']}尺寸。model.pt另附，与config一致。"+
                   '使用Python3.10、配套PyTorch/torchvision和requirements.txt；离线运行，依赖提前准备。'
                   f"训练入口{trained.get('training_source','optimized/train.py') if (run/'config.json').is_file() else 'optimized/train.py'}，需提供cache和相应初始权重作为--initial。"
                   'YOLO人物裁剪使用官方配套YOLO11n，权重在代码包中；开源许可证随包保留。'
                   '开发环境为torch2.9.1+cu128；官方Linux2.5.1+cu118尚待复验。')
    with zipfile.ZipFile(archive) as z:
        n, size = len(z.infolist()), sum(i.file_size for i in z.infolist())
        if n > 256 or size > 64*1024**2 or archive.stat().st_size > 20*1024**2:
            raise ValueError('Package exceeds competition limits')
    shutil.copy2(run/'best_ema.pt', dest/'model.pt')
    shutil.copy2(csv_path, dest/'validation.csv')
    write(dest/'metrics.json', {'run': str(a.run), 'metrics': report['metrics'], 'parameters': model.parameters,
          'architecture': config.get('architecture','r34'), 'components': config.get('components',[]),
          'flip_tta': a.flip, 'person_crop': a.crop, 'frames': config['frames'], 'latency_ms': float(np.mean(elapsed)),
          'weights_sha256': report['weights_sha256'], 'code_sha256': sha(archive),
          'report_file': (run/f'validation-{suffix}-report.html').relative_to(ROOT).as_posix(),
          'files': n, 'code_bytes': archive.stat().st_size, 'unpacked_bytes': size,
          'formal_submission_ready': False, 'linux_verified': False})
    # 新进程从实际代码ZIP加载，模拟组织方程序包与权重的组合。
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', YOLO_AUTOINSTALL='false')
    subprocess.run([sys.executable, str(ROOT/'tools/offline_probe.py'), '--package', str(archive),
                    '--weights', str(dest/'model.pt'), '--sample', str(ROOT/'data/raw/validation'/manifest['samples'][0]['file']),
                    '--output', str(dest/'offline-check.json')], check=True, env=env)
    print(json.dumps({'package': str(dest), 'metrics': report['metrics'], 'latency_ms': float(np.mean(elapsed)),
                      'parameters': model.parameters}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
