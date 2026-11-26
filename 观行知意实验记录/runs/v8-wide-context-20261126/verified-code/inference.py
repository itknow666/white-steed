import torch

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
