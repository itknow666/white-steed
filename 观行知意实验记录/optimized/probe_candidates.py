"""使用真实训练片段检查四种模型的严格预训练加载、梯度、保存重载和显存。"""
import argparse
import json
from pathlib import Path
import time
import torch
from candidate_models import Candidate, NAMES
from train_candidates import Clips

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=NAMES,required=True);p.add_argument('--batch',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    out=Path('runs/candidates-v6-20261106');out.mkdir(exist_ok=True)
    model=Candidate(a.name,f'weights/{a.name}-k400-v6.pth').cuda()
    ds=Clips('cache/crop16-192-v3','train')
    x=torch.stack([ds[i][0] for i in range(a.batch)]).cuda()
    y=torch.tensor([ds[i][1] for i in range(a.batch)],device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    torch.cuda.reset_peak_memory_stats();started=time.time()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y)
        assert logits.shape==(a.batch,40) and torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        assert norm>0
        opt.step()
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/2**20
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): reference=model(x).float().cpu()
    path=out/f'{a.name}-probe.pt';torch.save(model.state_dict(),path)
    del model,opt;torch.cuda.empty_cache()
    reloaded=Candidate(a.name).cuda().eval()
    reloaded.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): actual=reloaded(x).float().cpu()
    torch.testing.assert_close(reference,actual,rtol=1e-3,atol=1e-3)
    parameters=sum(v.numel() for v in reloaded.parameters())
    assert parameters+ds.meta['detector_parameters']<=100000000
    result=dict(name=a.name,passed=True,batch=a.batch,loss=float(loss.detach()),gradient_norm=float(norm),
                peak_mib=peak,seconds=time.time()-started,classifier_parameters=parameters,
                inference_parameters=parameters+ds.meta['detector_parameters'],real_train_samples=True,
                strict_pretrain_load=True,reload_match=True,torch=torch.__version__)
    (out/f'{a.name}-probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
