"""Cache raw depth/IR using fixed split manifests; no test data are read."""
import argparse,hashlib,json,time,re
from pathlib import Path
import numpy as np
from PIL import Image
from depth_ir_io import read_directories,frame_paths,uniform_indices

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def crop_window(boxes,w,h):
    b=np.asarray(boxes);x0,y0=b[:,:2].min(0);x1,y1=b[:,2:].max(0)
    cx,cy=(x0+x1)/2,(y0+y1)/2
    side=max(max(x1-x0,y1-y0)*1.4,.35*max(w,h))
    return [max(0,(cx-side/2)/w),max(0,(cy-side/2)/h),min(1,(cx+side/2)/w),min(1,(cy+side/2)/h)]

def source_resolution(sample):
    result={}
    for modality,key in [('Depth_Color','depth_dir'),('IR','ir_dir')]:
        with Image.open(frame_paths(sample[key])[0]) as image:
            result[modality]=list(image.size)
    return result

def reusable_crops(path,samples,detector_sha,detector_params):
    old=json.loads(Path(path).read_text(encoding='utf-8'))
    if (old.get('person_crop') is not True or old.get('detector_sha256')!=detector_sha
            or old.get('detector_parameters')!=detector_params):
        raise ValueError('Crop reuse requires the same detector weights and parameter count')
    rows=old['samples'];index={s['sample_id']:s for s in rows}
    if len(index)!=len(rows) or set(index)!={s['sample_id'] for s in samples}:
        raise ValueError('Crop reuse requires identical unique sample IDs')
    for sample in samples:
        previous=index[sample['sample_id']]
        for key in ('label','split','subject','published_manifest_sha256','published_clip_sha256'):
            if key not in sample or sample[key]!=previous.get(key):
                raise ValueError(f'Crop reuse metadata mismatch: {sample["sample_id"]} {key}')
        for key in ('published_manifest_sha256','published_clip_sha256'):
            if not isinstance(sample[key],str) or not re.fullmatch('[a-f0-9]{64}',sample[key]):
                raise ValueError('Crop reuse requires published content digests')
        current_size=source_resolution(sample)
        previous_size=previous.get('source_resolution') or source_resolution(previous)
        if current_size!=previous_size or current_size['Depth_Color']!=current_size['IR']:
            raise ValueError('Crop reuse source resolution mismatch')
        crop,boxes=previous.get('crop'),previous.get('detected_probe_boxes')
        if not isinstance(boxes,list) or len(boxes)>8 or (boxes and
                (np.asarray(boxes).shape!=(len(boxes),4) or not np.isfinite(boxes).all())):
            raise ValueError('Invalid cached person detection boxes')
        if crop is None:
            if boxes:raise ValueError('Missing crop despite detection boxes')
        else:
            if not boxes or len(crop)!=4 or not np.isfinite(crop).all() or not 0<=crop[0]<crop[2]<=1 or not 0<=crop[1]<crop[3]<=1:
                raise ValueError('Invalid cached crop window')
            if not np.allclose(crop,crop_window(boxes,*current_size['IR']),atol=1e-12,rtol=0):
                raise ValueError('Cached crop window differs from the fixed crop algorithm')
    return index

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--depth-mode',choices=['color','gray'],default='color')
    p.add_argument('--include-ir',action='store_true')
    p.add_argument('--person-crop',action='store_true')
    p.add_argument('--yolo-weights',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--reuse-crops',type=Path)
    a=p.parse_args();start=time.time()
    raw=json.loads(a.manifest.read_text(encoding='utf-8'))
    samples=raw['samples']
    assert set(s['split'] for s in samples)<= {'train','validation'}
    counts={sp:sum(s['split']==sp for s in samples) for sp in ('train','validation')}
    if not a.smoke: assert counts=={'train':1995,'validation':466},counts
    tr={str(s['subject']) for s in samples if s['split']=='train'}
    va={str(s['subject']) for s in samples if s['split']=='validation'}
    assert not tr&va,(tr,va)
    assert len({s['sample_id'] for s in samples})==len(samples)
    assert all(0<=int(s['label'])<40 for s in samples)
    a.output.mkdir(parents=True,exist_ok=False)
    model=None;detector_params=0
    if a.person_crop:
        from ultralytics import YOLO
        if not a.yolo_weights or not a.yolo_weights.is_file(): raise ValueError('local YOLO weights required')
        model=YOLO(str(a.yolo_weights));detector_params=sum(x.numel() for x in model.model.parameters())
    if a.reuse_crops and not model:raise ValueError('--reuse-crops requires --person-crop')
    reused=reusable_crops(a.reuse_crops,samples,sha(a.yolo_weights),detector_params) if a.reuse_crops else None
    ch=4 if a.include_ir else 3
    arrays={sp:np.lib.format.open_memmap(a.output/(sp+'.npy'),mode='w+',dtype=np.uint8,shape=(counts[sp],ch,a.frames,a.size,a.size)) for sp in counts}
    rows=[];idx={sp:0 for sp in counts};failures=[]
    for s in samples:
        depth=Path(s['depth_dir']);ir=Path(s['ir_dir']);win=None;boxes=[];probes=[]
        resolution=source_resolution(s)
        if reused is not None:
            previous=reused[s['sample_id']]
            win=previous['crop'];boxes=previous['detected_probe_boxes']
            paths=frame_paths(ir)
            probes=[str(paths[int(i)]) for i in sorted(set(uniform_indices(len(paths),8).tolist()))]
        elif model:
            paths=frame_paths(ir)
            if not paths: raise ValueError(f'missing IR localization: {s["sample_id"]}')
            probes=[str(paths[int(i)]) for i in sorted(set(uniform_indices(len(paths),8).tolist()))]
            for result in model.predict(probes,classes=[0],conf=.25,verbose=False,device=a.device):
                h,w=result.orig_shape
                if len(result.boxes): boxes.append(result.boxes.xyxy[result.boxes.conf.argmax()].tolist())
            if boxes: win=crop_window(boxes,w,h)
        try:
            x=read_directories(depth,ir,frames=a.frames,size=a.size,crop=win,depth_mode=a.depth_mode,include_ir=a.include_ir)
            if x.shape!=(ch,a.frames,a.size,a.size): raise ValueError(f'bad shape {x.shape}')
            if not np.isfinite(x).all(): raise ValueError('nonfinite input')
            if x.max()==0: raise ValueError('entire sample black')
        except Exception as e:
            failures.append({'sample_id':s['sample_id'],'error':str(e)})
            (a.output/'failures.json').write_text(json.dumps(failures,indent=2))
            raise
        sp=s['split'];arrays[sp][idx[sp]]=np.clip(np.rint(x*255),0,255).astype(np.uint8)
        rows.append(s|{'row':idx[sp],'crop':win,'detected_probe_boxes':boxes,'ir_probe_paths':probes,
            'source_resolution':resolution});idx[sp]+=1
        if len(rows)%50==0: print(json.dumps({'cached':len(rows),'total':len(samples),'seconds':time.time()-start}),flush=True)
    for ar in arrays.values(): ar.flush()
    result={'samples':rows,'counts':counts,'shape_per_sample':[ch,a.frames,a.size,a.size],
            'manifest_sha256':sha(a.manifest),'source_split_sha256':raw.get('source_split_sha256'),
            'depth_mode':a.depth_mode,'include_ir':a.include_ir,'person_crop':a.person_crop,
            'detector_parameters':detector_params,'detector_sha256':sha(a.yolo_weights) if model else None,
            'crop_fallback_count':sum(s['crop'] is None for s in rows) if model else None,
            'crop_reuse_manifest_sha256':sha(a.reuse_crops) if a.reuse_crops else None,
            'sampling':'independent uniform per modality, matching source baseline',
            'cache_seconds':time.time()-start,'test_accessed':False,'smoke':a.smoke}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)

if __name__=='__main__': main()
