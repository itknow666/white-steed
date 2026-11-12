"""Inference adapter matching raw-reproduction preprocessing and EMA weights."""
from pathlib import Path
import torch
from raw_model import RawModel
from depth_ir_io import sample_frame_lists, read_frame_lists, uniform_indices
from build_cache import crop_window


class Model:
    def __init__(self, config, device):
        self.config, self.device = config, device
        channels = int(config.get('channels', 4))
        self.model = RawModel(channels=channels, pretrained=None)
        weights = Path(config['weights_path'])
        if config.get('weights_format', 'pytorch') == 'safetensors':
            from safetensors.torch import load_file
            state = load_file(str(weights), device='cpu')
        else:
            state = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.detector = None
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if config.get('person_crop', False):
            from ultralytics import YOLO
            directory = Path(config.get('package_dir', Path(__file__).parent)).resolve()
            detector_path = (directory / config['detector_file']).resolve()
            if not detector_path.is_relative_to(directory) or not detector_path.is_file():
                raise ValueError('Package must contain the configured detector weights')
            self.detector = YOLO(str(detector_path))
            self.parameters += sum(p.numel() for p in self.detector.model.parameters())
        if self.parameters > 100_000_000:
            raise ValueError(f'Total inference parameters exceed 100M: {self.parameters}')

    @torch.inference_mode()
    def predict(self, sample_path):
        depth, ir = sample_frame_lists(sample_path)
        crop = None
        if self.detector is not None:
            probes = [str(ir[int(i)]) for i in sorted(set(uniform_indices(len(ir), 8).tolist()))]
            boxes = []
            for result in self.detector.predict(probes, classes=[0], conf=.25, verbose=False, device='cpu'):
                height, width = result.orig_shape
                if len(result.boxes):
                    boxes.append(result.boxes.xyxy[result.boxes.conf.argmax()].tolist())
            if boxes:
                crop = crop_window(boxes, width, height)
        value = read_frame_lists(depth, ir, frames=int(self.config.get('frames', 16)),
            size=int(self.config.get('size', 128)), crop=crop,
            depth_mode=self.config.get('depth_mode', 'color'), include_ir=self.config.get('channels', 4) == 4)
        tensor = torch.from_numpy(value).unsqueeze(0).to(self.device)
        return self.model(tensor)[0].float().cpu().tolist()
