"""Combine published train/validation manifests for the baseline cache builder."""
import argparse,hashlib,json
from pathlib import Path


def combine(data,output):
    rows=[];classes=None;subjects={};split_hashes={}
    for split in ('train','validation'):
        base=data/split;m=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
        published_manifest_sha=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()
        if m['modality']!='depth_ir' or len(m['classes'])!=40:raise ValueError('Expected the 40-class depth_ir dataset')
        if classes is not None and classes!=m['classes']:raise ValueError('Class order differs between splits')
        classes=m['classes'];subjects[split]=set()
        provenance_path=base/'provenance.json';provenance={}
        if provenance_path.exists():
            sidecar=json.loads(provenance_path.read_text(encoding='utf-8'))
            provenance={s['file']:s for s in sidecar['samples']}
            if len(provenance)!=len(sidecar['samples']) or set(provenance)!={s['file'] for s in m['samples']}:
                raise ValueError('Provenance samples differ from published manifest')
            split_hashes[split]=sidecar.get('source_split_sha256')
        for s in m['samples']:
            folder=(base/s['file']).resolve()
            if not folder.is_relative_to(base.resolve()):raise ValueError('Sample path escapes dataset')
            origin=provenance.get(s['file'],s)
            sid=origin.get('sample_id',s['file']+'.npy')
            if origin['label']!=s['label'] or sid!=s['file']+'.npy' or origin.get('split',split)!=split:
                raise ValueError('Provenance label, sample_id or split differs from manifest')
            source=origin.get('source_clip','')
            subject=origin.get('subject') or (source.split('/')[1] if len(source.split('/'))==3 else None)
            if not subject:raise ValueError('provenance.json requires subject or source_clip for split verification')
            subjects[split].add(str(subject))
            for mod in ('Depth_Color','IR'):
                if not (folder/mod).is_dir():raise ValueError(f'Missing {folder/mod}')
            rows.append({'sample_id':sid,'label':int(s['label']),
                'split':split,'subject':str(subject),'source_clip':source,
                'published_manifest_sha256':published_manifest_sha,'published_clip_sha256':s.get('sha256'),
                'depth_dir':str(folder/'Depth_Color'),'ir_dir':str(folder/'IR')})
    if subjects['train'] & subjects['validation']:raise ValueError('Train and validation subjects overlap')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate sample_id')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump({'samples':rows,'classes':classes,
        'source_split_sha256':split_hashes,'test_accessed':False},f,ensure_ascii=False,indent=2)
    return len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(combine(a.data,a.output))
