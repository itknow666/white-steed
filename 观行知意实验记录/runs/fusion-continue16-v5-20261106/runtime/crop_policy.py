"""可复现的扩框策略。仅改变定位后的边界，检测器和两个模态保持一致。"""
import numpy as np


def crop_window_scaled(boxes, width, height, scale=1.4):
    if not np.isfinite(scale) or scale < 1:
        raise ValueError('crop_scale must be finite and >= 1')
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError('Expected finite detection boxes')
    x0, y0 = boxes[:, :2].min(0)
    x1, y1 = boxes[:, 2:].max(0)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    side = max(max(x1-x0, y1-y0)*scale, .35*max(width, height))
    return [max(0, (cx-side/2)/width), max(0, (cy-side/2)/height),
            min(1, (cx+side/2)/width), min(1, (cy+side/2)/height)]
