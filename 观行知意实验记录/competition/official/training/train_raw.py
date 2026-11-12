"""Single-run reproduction of the public YOLO + R34 depth/IR training recipe."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from raw_model import RawModel

class CachedClips(Dataset):
    def __init__(self,root,split,training=False):
        self.root=root;self.split=split;self.training=training
        self.meta=json.loads((root/'manifest.json').read_text())
        self.rows=sorted([s for s in self.meta['samples'] if s['split']==split],key=lambda s:s['row'])
        self.array=None
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        if self.array is None: self.array=np.load(self.root/(self.split+'.npy'),mmap_mode='r')
        x=torch.from_numpy(self.array[i].copy()).float().div_(255)
        if self.training:
            if random.random()<.5: x=x.flip(-1)
            size=x.shape[-1];side=round(size*random.uniform(.75,1.))
            top=random.randint(0,size-side);left=random.randint(0,size-side)
            x=F.interpolate(x.permute(1,0,2,3)[:,:,top:top+side,left:left+side],size=(size,size),mode='bilinear',align_corners=False).permute(1,0,2,3)
        return x,int(self.rows[i]['label']),i

def metrics(labels,probs):
    cm=np.bincount(labels*40+probs.argmax(1),minlength=1600).reshape(40,40)
    tp=cm.diagonal()
    return {'accuracy':float(100*tp.sum()/len(labels)),
            'macro_f1':float(100*np.mean(2*tp/np.maximum(1,cm.sum(0)+cm.sum(1)))),
            'confusion':cm.tolist()}

@torch.inference_mode()
def evaluate(model,loader,device,tta=False):
    model.eval();probs=np.zeros((len(loader.dataset),40),np.float32);labels=np.zeros(len(loader.dataset),np.int64)
    for x,y,idx in loader:
        x=x.to(device,non_blocking=True)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
            pred=model(x).float().softmax(1)
            if tta: pred=(pred+model(x.flip(-1)).float().softmax(1))/2
        probs[idx.numpy()]=pred.cpu().numpy();labels[idx.numpy()]=y.numpy()
    return metrics(labels,probs),probs,labels

def write(path,obj): path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    tr=CachedClips(a.cache,'train',True);va=CachedClips(a.cache,'validation')
    if not a.smoke: assert len(tr)==1995 and len(va)==466 and not tr.meta.get('smoke')
    a.output.mkdir(parents=True,exist_ok=False)
    channels=tr.meta['shape_per_sample'][0];model=RawModel(channels,a.pretrained).to(a.device)
    params=sum(p.numel() for p in model.parameters());total=params+tr.meta['detector_parameters']
    assert total<=100_000_000,total
    meta={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
          'parameters':params,'inference_pipeline_parameters':total,'pretrained_sha256':digest(a.pretrained),
          'cache_manifest_sha256':digest(a.cache/'manifest.json'),'train_samples':len(tr),'validation_samples':len(va),
          'test_accessed':False,'selection':'last EMA epoch primary; best EMA reported separately',
          'recipe':'YOLO11n on8 IR probes, fixed union1.4 margin/.35 floor; R34 IG65M-K400; dropout .3; AdamW lr5e-5 wd1e-4; warmup2 cosine; EMA.99; same clip flip and crop',
          'gpu':torch.cuda.get_device_name() if a.device=='cuda' else 'cpu','torch':torch.__version__}
    write(a.output/'metadata.json',meta)
    if a.smoke:
        model.eval();x,y,_=tr[0];start=time.time()
        with torch.no_grad(): z=model(x.unsqueeze(0).to(a.device))
        assert z.shape==(1,40) and torch.isfinite(z).all()
        write(a.output/'smoke.json',{'shape':list(z.shape),'parameters':params,'total_parameters':total,'seconds':time.time()-start,'pretrained_loaded_strict':True})
        return
    from timm.utils import ModelEmaV3
    ema=ModelEmaV3(model,decay=.99,device=a.device)
    kwargs={'num_workers':a.workers,'pin_memory':a.device=='cuda','persistent_workers':a.workers>0}
    train=DataLoader(tr,batch_size=a.batch,shuffle=True,drop_last=True,**kwargs)
    val=DataLoader(va,batch_size=a.batch,shuffle=False,**kwargs)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=a.device=='cuda')
    total_steps=len(train)*a.epochs;warmup=len(train)*2;step=0;best=-1.;bestep=0;started=time.time()
    if a.device=='cuda': torch.backends.cudnn.benchmark=True;torch.cuda.reset_peak_memory_stats()
    ids=np.array([s['sample_id'] for s in va.rows])
    for ep in range(1,a.epochs+1):
        model.train();t0=time.time();loss_sum=0.;seen=0
        for x,y,_ in train:
            lr=a.lr*(step+1)/max(warmup,1) if step<warmup else a.lr*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for group in opt.param_groups: group['lr']=lr
            x=x.to(a.device,non_blocking=True);y=y.to(a.device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=a.device,dtype=torch.float16,enabled=a.device=='cuda'): loss=F.cross_entropy(model(x),y)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            norm=nn.utils.clip_grad_norm_(model.parameters(),5)
            # GradScaler skips an overflowed update and adjusts its scale,
            # matching the source fp16 recipe.
            scaler.step(opt);scaler.update();ema.update(model,step=step)
            step+=1;loss_sum+=loss.item()*len(y);seen+=len(y)
            if step==1: write(a.output/'pilot.json',{'loss':loss.item(),'grad_norm':norm.item(),'seconds':time.time()-started})
        train_seconds=time.time()-t0
        m,probs,labels=evaluate(ema.module,val,a.device)
        if m['accuracy']>best:
            best=m['accuracy'];bestep=ep;torch.save(ema.module.state_dict(),a.output/'best_ema.pt')
            write(a.output/'best_metrics.json',m|{'epoch':ep})
            np.savez_compressed(a.output/'best_predictions.npz',labels=labels,probabilities=probs,sample_ids=ids)
        row={'epoch':ep,'loss':loss_sum/seen,'train_loss':loss_sum/seen,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'val_acc':m['accuracy'],'val_f1':m['macro_f1'],'train_sec':train_seconds,'elapsed_sec':time.time()-started,'best_accuracy':best,'best_epoch':bestep,'lr':lr}
        with (a.output/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    elapsed=time.time()-started
    torch.save(ema.module.state_dict(),a.output/'last_ema.pt')
    write(a.output/'last_metrics.json',m|{'epoch':a.epochs})
    np.savez_compressed(a.output/'last_predictions.npz',labels=labels,probabilities=probs,sample_ids=ids)
    mt,pt,yt=evaluate(ema.module,val,a.device,tta=True)
    write(a.output/'last_flip_tta_metrics.json',mt)
    np.savez_compressed(a.output/'last_flip_tta_predictions.npz',labels=yt,probabilities=pt,sample_ids=ids)
    lat=[];sample=va[0][0].unsqueeze(0).to(a.device)
    with torch.inference_mode():
        for i in range(15):
            if a.device=='cuda': torch.cuda.synchronize()
            tt=time.perf_counter()
            with torch.autocast(device_type=a.device,dtype=torch.float16,enabled=a.device=='cuda'): ema.module(sample)
            if a.device=='cuda': torch.cuda.synchronize()
            if i>=5: lat.append((time.perf_counter()-tt)*1000)
    summary={'name':a.output.name,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'best_ema_accuracy':best,'last_flip_tta_accuracy':mt['accuracy'],
             'parameters':params,'inference_pipeline_parameters':total,'epochs':a.epochs,'train_eval_sec':elapsed,
             'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if a.device=='cuda' else None,
             'model_only_latency_ms_median':float(np.median(lat)),'cache_seconds':tr.meta['cache_seconds'],'test_accessed':False,
             'best_epoch':bestep,'selected_epoch':a.epochs,'stop_reason':'epoch_budget','checkpoint_bytes':(a.output/'last_ema.pt').stat().st_size}
    write(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
