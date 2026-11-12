"""下载四个通用 Kinetics400 预训练模型，记录实际文件哈希。"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request
from torchvision.models.video import MViT_V2_S_Weights

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'x3d_m': 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth',
    'tsm_r50': 'https://hanlab18.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth',
    'mvit_v2_s': MViT_V2_S_Weights.KINETICS400_V1.url,
    'uniformer_s': 'https://huggingface.co/Sense-X/uniformer_video/resolve/main/uniformer_small_k400_16x8.pth',
}

def download(item):
    name, url = item
    dest = ROOT/'weights'/f'{name}-k400-v6.pth'
    if not dest.exists():
        temp = dest.with_suffix('.partial')
        with urllib.request.urlopen(url, timeout=90) as src, temp.open('wb') as out:
            while b := src.read(1024*1024): out.write(b)
        temp.replace(dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    result = dict(name=name,url=url,path=str(dest),bytes=dest.stat().st_size,sha256=h,
                  training_data='generic Kinetics400 checkpoint; not CUHK-X',
                  use='initialization only; competition fine-tuning uses provided depth and IR')
    print(json.dumps(result),flush=True)
    return result

if __name__ == '__main__':
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(download, item): item[0] for item in FILES.items()}
        for job in concurrent.futures.as_completed(jobs):
            try: results.append(job.result())
            except Exception as e: results.append(dict(name=jobs[job],error=repr(e)))
    (ROOT/'competition/candidates-v6/weights-provenance.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
