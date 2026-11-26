from pathlib import Path
import hashlib
import json
import os
import sys
import zipfile

run = Path(__file__).resolve().parent
root = run.parents[1]
baseline = root / 'submission/v8-input-resolution-20261123/crop192-control'
dest = run / 'verified-wide-context'
code = dest / 'code'
code.mkdir(parents=True)
with zipfile.ZipFile(baseline / 'code.zip') as archive:
    archive.extractall(code)
(code / 'inference.py').rename(code / 'base_inference.py')
(code / 'inference.py').write_text("""import torch

from base_inference import Model as BaseModel


class Model(BaseModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.context_scale = float(config['context_scale'])
        self.context_weight = float(config['context_weight'])
        self.components = [{'crop_scale': self.context_scale}]

    @torch.inference_mode()
    def predict(self, sample_path):
        depth, ir, crops = self._paths_crop(sample_path, all_scales=True)
        standard = self._input(depth, ir, crops[float(self.config['crop_scale'])], self.config)
        context = self._input(depth, ir, crops[self.context_scale], self.config)
        main = self._scores(self.model, standard, False)
        if crops[self.context_scale] == crops[float(self.config['crop_scale'])]:
            return main.cpu().tolist()
        wide = self._scores(self.model, context, False)
        return ((1.0-self.context_weight)*main + self.context_weight*wide).cpu().tolist()
""", encoding='utf-8')
config = json.loads((code/'config.json').read_text(encoding='utf-8'))
config.update(context_scale=1.65, context_weight=.35)
(code/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
os.link(baseline/'model.pt', dest/'model.pt')
# The evaluator expects this path beside the extracted config.
os.link(dest/'model.pt', code/'model.pt')
with zipfile.ZipFile(dest/'code.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(code.rglob('*')):
        if path.is_file() and path != code/'model.pt':
            archive.write(path, path.relative_to(code).as_posix())
with zipfile.ZipFile(dest/'code.zip') as archive:
    assert archive.testzip() is None
    assert len(archive.infolist()) <= 256
    assert sum(info.file_size for info in archive.infolist()) <= 64*2**20
assert (dest/'code.zip').stat().st_size <= 20*2**20
sys.path.insert(0, str(root/'optimized'))
from package_domain_v8 import validate, sha
metrics, probe = validate(dest)
assert (dest/'validation.csv').read_bytes() == (run/'wider-context.csv').read_bytes()
report = {'metrics':metrics, 'parameters':probe['parameters'], 'offline_check':probe,
          'weights_sha256': sha(dest/'model.pt'), 'code_sha256': sha(dest/'code.zip')}
(dest/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
