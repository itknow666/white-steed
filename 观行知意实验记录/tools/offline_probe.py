"""在新进程中阻止Python网络连接，从实际ZIP加载模型并预测一个真实片段。

这是本地离线回归检查，不是操作系统沙箱或组织方Linux评测的替代品。
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import zipfile

import numpy as np
import torch


def denied(*args, **kwargs):
    raise RuntimeError('离线检查禁止联网')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['package', 'weights', 'sample', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    # 训练时下载预训练权重是允许的；推理必须只依赖本地最终权重。
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    with tempfile.TemporaryDirectory(prefix='wujie-offline-') as directory:
        root = Path(directory)
        with zipfile.ZipFile(args.package) as z:
            for member in z.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError('ZIP路径越界')
            z.extractall(root)
        config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        weights = root / config['weights_file']
        if not weights.resolve().is_relative_to(root):
            raise ValueError('权重路径越界')
        shutil.copy2(args.weights, weights)
        sys.path.insert(0, str(root))
        from inference import Model
        config.update(weights_path=str(weights), package_dir=str(root))
        model = Model(config, 'cuda' if torch.cuda.is_available() else 'cpu')
        scores = np.asarray(model.predict(args.sample.resolve()))
        if scores.shape != (40,) or not np.isfinite(scores).all():
            raise ValueError('推理接口结果无效')
        result = {'passed': True, 'python_network_connections_blocked': True,
                  'score_count': len(scores), 'finite': True, 'parameters': model.parameters,
                  'device': str(model.device), 'predicted_class': int(scores.argmax()),
                  'linux_verified': False, 'scope': 'Windows Python连接拦截回归检查；非系统级网络隔离'}
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
