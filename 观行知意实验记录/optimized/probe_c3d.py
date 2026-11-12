"""真实训练数据上的C3D前向、反向、重载及总参数检查。"""
from pathlib import Path
import json, torch
from c3d_model import Candidate
from train_c3d import Clips
torch.set_num_threads(4);torch.manual_seed(42)
ds=Clips('cache/crop16-v2','train')
x=torch.stack([ds[i][0] for i in range(4)]).cuda()
y=torch.tensor([ds[i][1] for i in range(4)],device='cuda')
m=Candidate('c3d','weights/c3d-sports1m-ucf101-openmmlab.pth').cuda()
opt=torch.optim.AdamW(m.parameters(),lr=1e-5)
torch.cuda.reset_peak_memory_stats()
for _ in range(2):
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  scores=m(x);loss=torch.nn.functional.cross_entropy(scores,y)
 assert scores.shape==(4,40) and torch.isfinite(loss)
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step()
peak=torch.cuda.max_memory_allocated()/2**20
params=sum(p.numel() for p in m.parameters())+ds.meta['detector_parameters'];assert params<=100_000_000
m.eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):before=m(x).cpu()
out=Path('runs/c3d-probe-20261110');out.mkdir(exist_ok=True)
torch.save(m.state_dict(),out/'smoke-only.pt')
del m,opt;torch.cuda.empty_cache()
m=Candidate('c3d').cuda().eval();m.load_state_dict(torch.load(out/'smoke-only.pt',map_location='cpu',weights_only=True),strict=True)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=m(x).cpu()
torch.testing.assert_close(before,after,rtol=.001,atol=.001)
result={'passed':True,'batch':4,'loss':float(loss.detach()),'grad_norm':float(norm),'peak_mib':peak,
        'inference_parameters_including_detector':params,'training_dataset':'4 samples, smoke only','reload_passed':True}
(out/'probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
