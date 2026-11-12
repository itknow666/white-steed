"""Recover exact published clip IDs from the organizer's original SHA256 rule."""
import argparse,hashlib,json,re,io
from pathlib import Path
import numpy as np
from PIL import Image

EXPECTED={'train':'b1da519efcaffea98e2b153f39a3195bfa902617826135e866bbca289380ed28','validation':'6e5c15ff25b726d4925c4e58820204746db83551e10156a303571c0d84b6baa5'}
def digest(x): return hashlib.sha256(x).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--depth',type=Path,required=True);p.add_argument('--ir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pixel-check-count',type=int,default=20)
    p.add_argument('--allow-missing-ir',action='store_true');a=p.parse_args()
    lookup={};manifests={}
    for split,expected in EXPECTED.items():
        b=(a.prepared/split/'manifest.json').read_bytes();assert digest(b)==expected
        manifests[split]=json.loads(b)
        for s in manifests[split]['samples']:
            assert s['file'] not in lookup
            lookup[s['file']]=s|{'split':split}
    found={}
    for trial in a.depth.glob('*/*/*'):
        if not trial.is_dir(): continue
        relative=trial.relative_to(a.depth).as_posix();sample_id=digest(relative.encode())[:32]+'.npy'
        if sample_id not in lookup: continue
        source=lookup[sample_id];action,subject,take=relative.split('/')
        assert int(action.split('_')[0])==source['label']
        assert sample_id not in found
        ir=a.ir/relative
        if not a.allow_missing_ir: assert ir.is_dir() and any(ir.glob('*.png')),str(ir)
        found[sample_id]={'sample_id':sample_id,'label':source['label'],'split':source['split'],'subject':subject,
                          'source_clip':relative,'depth_dir':str(trial),'ir_dir':str(ir)}
    missing=sorted(set(lookup)-set(found));assert not missing,{'missing_count':len(missing),'examples':missing[:10]}
    samples=[found[s['file']] for split in EXPECTED for s in manifests[split]['samples']]
    checks=[]
    for split in EXPECTED:
        rows=[s for s in samples if s['split']==split]
        for i in np.linspace(0,len(rows)-1,min(a.pixel_check_count,len(rows))).astype(int):
            s=rows[i];paths=list(Path(s['depth_dir']).glob('*.png'))
            paths.sort(key=lambda x:int(re.search(r'_([0-9]+)_Color\.png$',x.name)[1]))
            xs=[]
            for index in np.linspace(0,len(paths)-1,16).astype(int):
                with Image.open(paths[index]) as im: xs.append(np.asarray(im.convert('L').resize((112,112),Image.Resampling.BILINEAR),dtype=np.float32)/255)
            x=np.stack(xs)[:,None]
            old=np.load(a.prepared/split/s['sample_id'],allow_pickle=False)
            equal=np.array_equal(x,old);maxdiff=float(np.abs(x-old).max())
            checks.append({'sample_id':s['sample_id'],'split':split,'equal':equal,'max_abs_difference':maxdiff})
            assert equal,checks[-1]
    result={'samples':samples,'source_split_sha256':EXPECTED,'id_rule':'SHA256(action/user/trial)[:32]+.npy, exact organizer prepare_cuhkx_depth.py rule',
            'pixel_checks':checks,'test_accessed':False,'ir_required':not a.allow_missing_ir}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'samples':len(samples),'pixel_checks':len(checks),'all_exact':all(x['equal'] for x in checks)}))

if __name__=='__main__':main()
