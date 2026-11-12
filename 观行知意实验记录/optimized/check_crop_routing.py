"""用实际图片和独立缓存验证融合时两个分类器分别得到正确窗口。

检测结果注入已有缓存的IR框，避免与正在训练的GPU争用；完整检测在最终复验执行。
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'baseline'))
sys.path.insert(0,str(Path(__file__).parent))
from inference import Model
from train import write


class Boxes:
    def __init__(self,box):
        self.xyxy=torch.tensor([box],dtype=torch.float64)
        self.conf=torch.ones(1)
    def __len__(self): return 1


class FixedDetector:
    def __init__(self,boxes,shape): self.boxes,self.shape=boxes,shape
    def predict(self,*args,**kwargs):
        return [SimpleNamespace(orig_shape=self.shape,boxes=Boxes(box)) for box in self.boxes]


def main():
    old=json.loads((ROOT/'cache/crop16-v2/manifest.json').read_text(encoding='utf-8'))
    rows=[r for r in old['samples'] if r['split']=='validation']
    arrays={1.4:np.load(ROOT/'cache/crop16-v2/validation.npy',mmap_mode='r'),
            1.8:np.load(ROOT/'cache/wide16-v4-20261027/validation.npy',mmap_mode='r')}
    model=Model.__new__(Model)
    model.config={'crop_scale':1.8};model.device='cpu'
    model.components=[{'crop_scale':1.8},{'crop_scale':1.4}]
    checks=[]
    selected=[rows[0],rows[len(rows)//2],rows[-1],next(r for r in rows if not r['detected_probe_boxes'])]
    for row in selected:
        width,height=row['source_resolution']['IR']
        model.detector=FixedDetector(row['detected_probe_boxes'],(height,width))
        depth,ir,crops=model._paths_crop(Path(row['depth_dir']).parent,all_scales=True)
        for scale,array in arrays.items():
            x=model._input(depth,ir,crops[scale],{'frames':16,'size':128})
            error=float(np.max(np.abs(x-array[row['row']].astype(np.float32)/255)))
            if error>1e-7: raise AssertionError(f'Wrong crop routed: {scale}, error {error}')
            checks.append({'sample_id':row['sample_id'],'crop_scale':scale,'max_abs_error':error})
    result={'passed':True,'checks':checks,'scope':'Real images with cached IR detections injected; actual detector checked at final evaluation'}
    write(ROOT/'runs/ablation-v4-20261027/crop-routing-check.json',result)
    print(json.dumps(result))


if __name__=='__main__': main()
