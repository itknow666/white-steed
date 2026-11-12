"""验证阶段边界：冻结时无梯度/BN变化，解冻后首层重新获得梯度。"""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).parent))
from train_ablation import RawModel, set_stage, write


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=RawModel(channels=4,dropout=.5)
    model.train()
    count_frozen=set_stage(model,True)
    first=model.encoder.stem[0].weight
    bn=next(m for m in model.encoder.stem.modules() if isinstance(m,torch.nn.BatchNorm3d))
    mean=bn.running_mean.clone()
    x=torch.rand(2,4,4,32,32)
    model(x).square().mean().backward()
    assert first.grad is None and torch.equal(bn.running_mean,mean)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
    model.zero_grad(set_to_none=True)
    model.train()
    count_full=set_stage(model,False)
    model(x).square().mean().backward()
    assert first.grad is not None and torch.isfinite(first.grad).all() and first.grad.abs().sum()>0
    assert not torch.equal(bn.running_mean,mean)
    result={'frozen_stem_has_no_gradient':True,'frozen_bn_unchanged':True,
            'head_has_finite_gradient':True,'unfrozen_stem_has_finite_nonzero_gradient':True,
            'unfrozen_bn_updates':True,'trainable_frozen':count_frozen,'trainable_full':count_full,
            'device':'cpu','purpose':'stage boundary mechanics, not accuracy evidence'}
    write(Path(__file__).resolve().parents[1]/'runs/ablation-v4-20261027/stage-check.json',result)
    print(json.dumps(result))


if __name__=='__main__': main()
