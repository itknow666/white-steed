"""与赛事Model接口兼容；只加载本地参数，支持同步人物裁剪及翻转平均。"""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from raw_model import RawModel
from depth_ir_io import sample_frame_lists, read_frame_lists, uniform_indices
from build_cache import crop_window


class Model:
    def __init__(self, config, device):
        self.config, self.device = dict(config), str(device)
        architecture = config.get('architecture', 'r34')
        self.components = config.get('components', []) if architecture == 'ensemble' else []
        if architecture == 'ensemble':
            if not 2 <= len(self.components) <= 3:
                raise ValueError('Expected two or three explicitly configured components')
            self.model = torch.nn.ModuleList([self._classifier(s['architecture']) for s in self.components])
            weights = [float(s['weight']) for s in self.components]
            if any(not torch.isfinite(torch.tensor(w)) or w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError('Invalid component weights')
        elif architecture == 'swin3dt':
            from video_swin import VideoSwin
            self.model = VideoSwin()
        elif architecture in ['x3d_m','tsm_r50','mvit_v2_s','uniformer_s']:
            self.model = self._classifier(architecture)
        else:
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

    @staticmethod
    def _classifier(name):
        if name == 'r34':
            return RawModel(channels=4,pretrained=None)
        if name == 'swin3dt':
            from video_swin import VideoSwin
            return VideoSwin()
        from candidate_models import Candidate
        return Candidate(name)

    @torch.inference_mode()
    def _paths_crop(self, sample_path, all_scales=False):
        depth, ir = sample_frame_lists(sample_path)
        scales = {float(self.config.get('crop_scale',1.4))}
        if all_scales:
            scales.update(float(s.get('crop_scale',1.4)) for s in self.components)
        if any(not np.isfinite(s) or s < 1 for s in scales):
            raise ValueError('Invalid crop scale')
        crops = {s:None for s in scales}
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
                for scale in scales:
                    if scale == 1.4:
                        crops[scale] = crop_window(boxes, width, height)
                    else:
                        from crop_policy import crop_window_scaled
                        crops[scale] = crop_window_scaled(boxes,width,height,scale)
        crop = crops if all_scales else crops[float(self.config.get('crop_scale',1.4))]
        return depth, ir, crop

    @torch.inference_mode()
    def prepare(self, sample_path):
        depth, ir, crop = self._paths_crop(sample_path)
        return self._input(depth, ir, crop, self.config)

    def _input(self, depth, ir, crop, spec):
        x = read_frame_lists(depth, ir, frames=int(spec.get('frames',16)), size=int(spec.get('size',128)),
                             crop=crop, depth_mode='color', include_ir=True)
        if spec.get('ir_contrast',False):
            from ir_contrast import enhance
            x = enhance(np.rint(x*255).astype(np.uint8)).astype(np.float32)/255.
        return x

    @torch.inference_mode()
    def predict(self, sample_path):
        if self.components:
            # 两个分类器共享同一次人物定位；全部参数在初始化时统一计数。
            depth, ir, crops = self._paths_crop(sample_path, all_scales=True)
            scores, total = None, 0.
            for classifier, spec in zip(self.model, self.components):
                x = self._input(depth, ir, crops[float(spec.get('crop_scale',1.4))], spec)
                value = self._scores(classifier, x, bool(spec.get('flip_tta', False)))
                weight = float(spec['weight'])
                scores = value*weight if scores is None else scores+value*weight
                total += weight
            return (scores/total).cpu().tolist()
        x = self.prepare(sample_path)
        return self._scores(self.model, x, bool(self.config.get('flip_tta', False))).cpu().tolist()

    def _scores(self, classifier, x, flip):
        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        if flip:
            tensor = torch.cat([tensor, tensor.flip(-1)], dim=0)
        if self.device.startswith('cuda') and self.config.get('channels_last_3d', True):
            tensor = tensor.contiguous(memory_format=torch.channels_last_3d)
        context = torch.autocast('cuda', dtype=torch.float16) if self.device.startswith('cuda') and self.config.get('amp', True) else nullcontext()
        with context:
            logits = classifier(tensor)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite output')
        # 两个视角共享同一模型，不增加参数；概率平均须与训练验证保持一致。
        return logits.float().softmax(1).mean(0)
