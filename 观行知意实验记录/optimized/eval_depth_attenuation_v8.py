"""Evaluate partial depth attenuation and clean warm-start recovery."""
import argparse
from pathlib import Path

import torch

from cache_teacher_v8 import training_logits
from eval_modality_dropout_v8 import modality_probe
from eval_phase_order_v8 import RAW, compare, read
from eval_sensor_robustness_v8 import BASE
from eval_shared_raw_v8 import shared_raw
from fit_frozen_head_v8 import official_split
from package_domain_v8 import ROOT
from train_videomae import metrics, save, sha, write
from videomae_v8 import DomainVideoMAE


TRAINED = ('attenuation-fixed','attenuation-ramp','dropout-recovery')
BLEND = 'adapter-interpolation'
ARMS = dict.fromkeys((*TRAINED,BLEND),'videomae2_dual_v8')
COLUMNS = ('固定轻度削弱','逐轮恢复深度','从隐藏模态模型恢复','原适配器加隐藏模态增量')


def adapter_endpoints():
    source = ROOT/'runs/v8-modality-dropout-20261123'
    paths = [source/name/'best_ema.pt' for name in ('adapter-control','adapter-moddrop')]
    control, dropped = [torch.load(path,weights_only=True,map_location='cpu') for path in paths]
    prefixes = ('encoder.patch_embed.depth_adapter.','encoder.patch_embed.ir_adapter.')
    names = {key for key in control if key.startswith(prefixes)}
    if (control.keys()!=dropped.keys() or not names
            or any(not torch.equal(control[key],dropped[key]) for key in control.keys()-names)):
        raise ValueError('Only input adapter tensors may differ between interpolation endpoints')
    return paths, control, dropped, names


def adapter_state(control, dropped, names, weight):
    return {key:torch.lerp(v,dropped[key],weight) if key in names else v for key,v in control.items()}


def interpolate_adapters(runs):
    evidence = runs/'interpolation-diagnostic.json'
    if evidence.is_file():
        return
    paths, control, dropped, names = adapter_endpoints()
    data, _, labels = official_split('validation',466)
    folder = runs/BLEND
    folder.mkdir(exist_ok=True)
    temporary = folder/'preview.pt'
    candidates = []
    for weight in (.25,.50,.75):
        state = adapter_state(control,dropped,names,weight)
        save(temporary,state)
        row = {}
        for name,variant in (('dual','plain'),('ir-only','ironly')):
            scores = training_logits(DomainVideoMAE(variant=variant),data,temporary)
            result = metrics(labels,scores)
            result.pop('confusion')
            row[name] = result
        candidates.append(dict(weight=weight,**row))
        print(weight,row,flush=True)
    best = max(candidates,key=lambda row:(row['dual']['accuracy'],row['dual']['macro_f1'],row['ir-only']['accuracy']))
    weight = best['weight']
    chosen = adapter_state(control,dropped,names,weight)
    save(folder/'best_ema.pt',chosen)
    temporary.unlink()
    hashes = [sha(path) for path in paths]
    write(folder/'config.json',{'original_control_sha256':hashes[0],'source_moddrop_sha256':hashes[1],
                                'moddrop_fraction':weight})
    write(folder/'summary.json',{'mode':BLEND,'training_steps':0})
    write(evidence,{'samples':len(labels),'candidates':candidates,'selected_fraction':weight})


def stage_ir_interpolation(runs):
    name = 'adapter-interpolation-75'
    folder = runs/name
    if not (folder/'summary.json').is_file():
        paths, control, dropped, names = adapter_endpoints()
        folder.mkdir(parents=True,exist_ok=True)
        save(folder/'best_ema.pt',adapter_state(control,dropped,names,.75))
        write(folder/'config.json',{'original_control_sha256':sha(paths[0]),
                                    'source_moddrop_sha256':sha(paths[1]),'moddrop_fraction':.75})
        write(folder/'summary.json',{'mode':name,'training_steps':0})
    return {name:'videomae2_dual_v8'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--ir-tilt',action='store_true',help='Confirm the 75% blend on original images')
    args=parser.parse_args()
    torch.set_num_threads(4)
    runs,out=args.runs.resolve(),args.output.resolve()
    manifest=read(RAW/'manifest.json')
    if args.ir_tilt:
        arms = stage_ir_interpolation(runs)
        columns = ('隐藏组系数 0.75',)
    else:
        interpolate_adapters(runs)
        arms, columns = ARMS, COLUMNS
    shared_raw(runs,out,RAW,manifest,arms,BASE)
    compare(runs,out,manifest,arms=arms,columns=columns,title='深度特征部分削弱与双路恢复',baseline=BASE)
    modality_probe(runs,arms)


if __name__=='__main__':
    main()
