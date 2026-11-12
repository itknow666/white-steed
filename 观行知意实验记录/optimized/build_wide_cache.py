"""复用已经校验的IR检测框，从原图生成扩大裁剪缓存，不重新检测、不改标签。"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'baseline'))
from build_cache import reusable_crops, crop_window
from depth_ir_io import read_directories, frame_paths
from crop_policy import crop_window_scaled
from train import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scale', type=float, default=1.8)
    a = p.parse_args()
    started = time.time()
    old = json.loads((a.source/'manifest.json').read_text(encoding='utf-8'))
    if old['counts'] != {'train':1995, 'validation':466} or old['smoke']:
        raise ValueError('Full published split required')
    reusable_crops(a.source/'manifest.json', old['samples'], old['detector_sha256'], old['detector_parameters'])
    if sha(ROOT/'baseline/yolo11n.pt') != old['detector_sha256']:
        raise ValueError('Detector source mismatch')
    a.output.mkdir(parents=True, exist_ok=False)
    meta = copy.deepcopy(old)
    arrays = {sp: np.lib.format.open_memmap(a.output/(sp+'.npy'), mode='w+', dtype=np.uint8,
               shape=(n,*old['shape_per_sample'])) for sp,n in old['counts'].items()}
    examples, areas = [], []
    for i,row in enumerate(meta['samples']):
        boxes = row['detected_probe_boxes']
        w,h = row['source_resolution']['IR']
        original = row['crop']
        win = crop_window_scaled(boxes,w,h,a.scale) if boxes else None
        if boxes:
            if not np.allclose(crop_window_scaled(boxes,w,h,1.4),crop_window(boxes,w,h),atol=1e-12,rtol=0):
                raise AssertionError('Default policy must match original')
            if not (win[0] <= original[0] and win[1] <= original[1] and win[2] >= original[2] and win[3] >= original[3]):
                raise AssertionError('Enlarged crop must contain old crop')
            areas.append((win[2]-win[0])*(win[3]-win[1])/((original[2]-original[0])*(original[3]-original[1])))
        row['crop'] = win
        x = read_directories(row['depth_dir'],row['ir_dir'],frames=16,size=128,crop=win,depth_mode='color',include_ir=True)
        if x.shape != (4,16,128,128) or not np.isfinite(x).all():
            raise ValueError('Invalid cached clip')
        arrays[row['split']][row['row']] = np.rint(x*255).clip(0,255).astype(np.uint8)
        # 固定、按类别和ID选择画面；人工观察不参与参数学习。
        if row['split']=='validation' and row['label'] not in [e[0] for e in examples] and len(examples)<12:
            paths = frame_paths(row['ir_dir'])
            image = Image.open(paths[len(paths)//2]).convert('RGB')
            draw = ImageDraw.Draw(image)
            for box,color in [(original,'red'),(win,'lime')]:
                if box:
                    draw.rectangle([round(box[0]*w),round(box[1]*h),round(box[2]*w)-1,round(box[3]*h)-1],outline=color,width=2)
            examples.append((row['label'],image,row['sample_id']))
        if (i+1)%200==0:
            print(json.dumps({'cached':i+1,'total':2461,'seconds':time.time()-started}),flush=True)
    for array in arrays.values(): array.flush()
    meta.update(crop_scale=a.scale,crop_reuse_manifest_sha256=sha(a.source/'manifest.json'),cache_seconds=time.time()-started,
                crop_algorithm='optimized/crop_policy.py:crop_window_scaled',test_accessed=False)
    write(a.output/'manifest.json',meta)
    canvas = Image.new('RGB',(304*4,250*3),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(label,image,sample_id) in enumerate(examples):
        x,y = i%4*304,i//4*250
        canvas.paste(image,(x,y));draw.text((x+2,y+229),f'class {label}: red=1.4 green={a.scale}',fill='black')
    canvas.save(a.output/'crop-comparison.png')
    audit = {'samples':2461,'fallbacks':meta['crop_fallback_count'],'crop_scale':a.scale,
             'mean_area_ratio':float(np.mean(areas)),'median_area_ratio':float(np.median(areas)),
             'default_matches_old':True,'all_wide_contain_old':True,'test_accessed':False}
    write(a.output/'audit.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__': main()
