"""片段级IR百分位拉伸：只用当前片段的像素，不拟合验证/测试标签。"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np


def enhance(clip):
    """输入/输出uint8 [4,T,H,W]。深度不变，全段IR共用一个亮度映射。"""
    if clip.dtype != np.uint8 or clip.ndim != 4 or clip.shape[0] != 4:
        raise ValueError('Expected uint8 [4,T,H,W]')
    ir = clip[3]
    histogram = np.bincount(ir.reshape(-1), minlength=256)
    cdf = np.cumsum(histogram)
    low, high = [int(np.searchsorted(cdf, ir.size*q)) for q in [.01,.99]]
    result = clip.copy()
    # 避免近乎全黑的片段被无限放大噪声；不改变动作时间顺序。
    span = max(8, high-low)
    table = np.clip(np.rint((np.arange(256)-low)*255./span),0,255).astype(np.uint8)
    result[3] = table[ir]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous cache')
    a.output.mkdir()
    m = json.loads((a.source/'manifest.json').read_text(encoding='utf-8'))
    if m.get('ir_contrast'):
        raise ValueError('Already enhanced')
    started = time.time()
    for split in ['train','validation']:
        old = np.load(a.source/(split+'.npy'),mmap_mode='r')
        out = np.lib.format.open_memmap(a.output/(split+'.npy'),mode='w+',dtype=np.uint8,shape=old.shape)
        for i in range(len(old)):
            out[i]=enhance(old[i])
        out.flush();old._mmap.close();out._mmap.close()
        print(json.dumps({'split':split,'samples':len([s for s in m['samples'] if s['split']==split])}),flush=True)
    m.update(ir_contrast=True,ir_contrast_method='per-clip IR 1/99 histogram percentiles, span>=8, round uint8',
             source_cache_sha256=hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),
             contrast_cache_seconds=time.time()-started)
    (a.output/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
