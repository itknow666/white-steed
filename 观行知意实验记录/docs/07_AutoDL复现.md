# AutoDL与Linux参考环境复现

本机开发已获授权；本次不租用GPU、不消耗算力券。你们建立实例后，用平台提供的SSH配置在VS Code Remote SSH中连接，或使用平台终端。上传项目源码、原始数据、最终权重；不要上传Windows的.venv或environment/python。先阅读实例收费规则并在结束后按平台要求停止实例。

## 两套环境

| 环境 | Python | PyTorch / torchvision | 用途 |
|---|---|---|---|
| 本机RTX5070Ti | 3.10独立安装 | 2.9.1+cu128 / 0.24.1+cu128 | Windows GPU开发 |
| 官方参考 | 3.10（参考3.10.20） | 2.5.1+cu118 / 0.20.1+cu118 | 与组织方接近的Linux兼容检查 |

官方其余版本在environment/official-reference.json。默认不启用人物检测，仅安装numpy、Pillow、timm等基础依赖；ultralytics/opencv是可选裁剪相关，本基线不需要。若将来启用新依赖，提前登记。

选择能够支持官方CUDA11.8构建的云端GPU镜像。Blackwell显卡不可直接假定能运行旧构建。若平台镜像不兼容，应联系组织方确认测试环境，不能悄悄把新环境声称为官方环境。

```bash
cd /path/to/wujie-action
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -r environment/requirements-base.txt
.venv/bin/python tools/project.py check-env
# Windows生成的完整缓存元数据包含绝对路径，跨平台后应重新生成清单/缓存。
# 将旧cache目录另存，不要沿用Windows目录字符串。
.venv/bin/python tools/project.py prepare --small
```

已有最终权重时，上传对应runs目录下的inference-config.json、metadata.json和.pt。运行：

```bash
.venv/bin/python tools/project.py predict --run runs/full16
.venv/bin/python tools/project.py package --run runs/full16 --name linux-final
```

工具的linux_verified字段默认保守为false；真实运行成功后，在验收记录注明实例GPU、系统、版本、命令、耗时和输出SHA256，并另存Linux记录，不手工改造未验证的结果。注意断网验证不依赖预训练文件，只有训练初始化才需要它。

如需重训：先prepare完整缓存再train。开始30轮前先做短测估计每轮时间，结合100元券预算决定实验。推荐先做单模型，先看错例再增加训练成本。
