"""与赛事Model接口兼容；只加载本地参数，支持同步人物裁剪及翻转平均。"""
from contextlib import nullcontext
from pathlib import Path
import torch
from raw_model import RawModel
from depth_ir_io import sample_frame_lists, read_frame_lists, uniform_indices
from build_cache import crop_window


class Model:
    def __init__(self, config, device):
        self.config, self.device = dict(config), str(device)
        self.model = RawModel(channels=int(config.get('channels', 4)), pretrained=None)
        self.model.load_state_dict(torch.load(config['weights_path'], map_location='cpu', weights_only=True), strict=True)
        self.model.to(device).eval()
        if self.device.startswith('cuda') and config.get('channels_last_3d', True):
            self.model.to(memory_format=torch.channels_last_3d)
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.detector = None
        if config.get('person_crop', False):
            from ultralytics import YOLO
            directory = Path(config.get('package_dir', Path(__file__).parent)).resolve()
            detector_path = (directory/config['detector_file']).resolve()
            if not detector_path.is_relative_to(directory) or not detector_path.is_file():
                raise ValueError('Missing local detector file')
            self.detector = YOLO(str(detector_path))
            self.parameters += sum(p.numel() for p in self.detector.model.parameters())
        if self.parameters > 100_000_000:
            raise ValueError('Inference parameters exceed 100M')

    @torch.inference_mode()
    def prepare(self, sample_path):
        depth, ir = sample_frame_lists(sample_path)
        crop = None
        if self.detector is not None:
            probes = [str(ir[int(i)]) for i in sorted(set(uniform_indices(len(ir), 8).tolist()))]
            boxes = []
            # 与官方缓存生成器相同的conf、imgsz和裁剪算法；GPU仅加快运算。
            for result in self.detector.predict(probes, classes=[0], conf=.25, imgsz=640,
                    verbose=False, device=self.device):
                height, width = result.orig_shape
                if len(result.boxes):
                    boxes.append(result.boxes.xyxy[result.boxes.conf.argmax()].tolist())
            if boxes:
                crop = crop_window(boxes, width, height)
        return read_frame_lists(depth, ir, frames=int(self.config.get('frames', 16)),
                size=int(self.config.get('size', 128)), crop=crop, depth_mode='color', include_ir=True)

    @torch.inference_mode()
    def predict(self, sample_path):
        x = self.prepare(sample_path)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        flip = bool(self.config.get('flip_tta', False))
        if flip:
            tensor = torch.cat([tensor, tensor.flip(-1)], dim=0)
        if self.device.startswith('cuda') and self.config.get('channels_last_3d', True):
            tensor = tensor.contiguous(memory_format=torch.channels_last_3d)
        context = torch.autocast('cuda', dtype=torch.float16) if self.device.startswith('cuda') and self.config.get('amp', True) else nullcontext()
        with context:
            logits = self.model(tensor)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite output')
        # 两个视角共享同一模型，不增加参数；概率平均须与训练验证保持一致。
        return logits.float().softmax(1).mean(0).cpu().tolist()
