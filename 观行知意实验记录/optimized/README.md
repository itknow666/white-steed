# 本机优化实验

从项目根目录 `<项目目录>` 执行命令，使用项目的 `.venv`。最终选择、实际指标和运行方法见 `runs/优化训练结果_20261025.md`；完整逐轮记录在各实验目录的 `history.jsonl`。

## 本次方法

保留官方 R(2+1)D-34 结构和四通道深度/IR输入，用本团队初版权重继续微调。`train.py` 包含中文说明：BF16、三维 channels-last 布局、骨干/分类头不同学习率、慢速 BN 统计、标签平滑、温和类别加权、EMA和早停。类别权重只统计训练集；验证集只用于评估和选择。

人物裁剪复用官方开发包的 YOLO11n 定位与裁剪算法，在 IR 上检测，给同步的深度图和 IR 使用同一个框。不是使用额外 RGB 数据。模型来源和许可证见 `competition/optimization-v2-sources.json`。

## 重现实验

缓存较大但可以重新生成。已有 `cache/full16`、`cache/crop16-v2` 和 `cache/crop32-v2` 可直接复用；不要覆盖原始数据划分。以下输出路径故意使用新名字，保留已完成的实验：

```powershell
# 16帧、不裁剪的对照：最多12轮，验证停滞时提前停止。
.\.venv\Scripts\python.exe optimized/train.py --cache cache/full16 --output runs/repeat-stable16 --initial runs/initial16-20261020-5ep/best_ema.pt --epochs 12 --batch 8

# 16帧、人物裁剪：同样从本团队初版权重开始。
.\.venv\Scripts\python.exe optimized/train.py --cache cache/crop16-v2 --output runs/repeat-crop16 --initial runs/initial16-20261020-5ep/best_ema.pt --epochs 12 --batch 8

# 32帧：从已经完成的16帧裁剪模型继续，降低学习率。
.\.venv\Scripts\python.exe optimized/train.py --cache cache/crop32-v2 --output runs/repeat-crop32 --initial runs/crop16-v2-20261025/best_ema.pt --epochs 8 --batch 4 --lr 0.00001 --patience 3
```

运行前核对初版实际权重路径；精确启动参数以各已完成实验的 `config.json` 为准。中断恢复使用相同输出目录加 `--resume`，恢复模型、EMA、优化器和随机数状态。不要仅加载模型冒充完整恢复。

```powershell
.\.venv\Scripts\python.exe optimized/train.py --cache cache/crop16-v2 --output runs/repeat-crop16 --initial runs/initial16-20261020-5ep/best_ema.pt --epochs 12 --batch 8 --resume

# 从真实图片逐段推理、生成466行CSV、分类分析，并打包和进行离线加载检查。
.\.venv\Scripts\python.exe optimized/evaluate.py --run runs/repeat-crop16 --package-name repeat-crop16-single --crop --check-cache cache/crop16-v2
```

`--flip` 让同一模型处理原图及水平翻转，平均分数；是否有益由验证结果决定。`--input-frames 32` 仅改变推理取帧，不更新参数，用于分开检查训练和推理帧数的影响。多个完整分类模型相加可能超过比赛100M参数限制，本项目没有这样做。

`finalize.py` 是本次日期固定的一次性交付整理工具。它会选择已完成的独立验证结果并创建 `submission/optimized-20261025`，不要重复运行覆盖交付目录。后续新实验应使用新名称单独保存。

## 环境与结果边界

继续探索的训练器是 `train_generalize.py`（前三组R34路线）和 `train_swin.py`（TorchVision Video Swin-T）。使用相同骨架保留原训练器，避免修改已完成实验：

```powershell
# 例：时序增强组；训练从32帧缓存随机选16帧，验证用固定16帧缓存。
.\.venv\Scripts\python.exe optimized/train_generalize.py --cache cache/crop32-v2 --val-cache cache/crop16-v2 --frames 16 --output runs/repeat-temporal16 --initial runs/crop16-v2-20261025/best_ema.pt --epochs 8 --batch 8 --lr 0.00001 --head-multiplier 5 --freeze-bn --temporal-jitter --mixup-alpha 0.2 --erase-prob 0.1 --patience 3

# 例：不同架构；公开预训练在训练前下载并校验，推理不下载。
.\.venv\Scripts\python.exe optimized/train_swin.py --cache cache/crop16-192-v3 --output runs/repeat-swin192 --initial weights/swin3d_t-7615ae03.pth --epochs 12 --batch 4 --lr 0.0001 --mixup-alpha 0.2 --erase-prob 0.1 --patience 4 --min-epochs 6
```

`compare_views.py` 在缓存上快速检查单视角/翻转平均；最终仍需 `evaluate.py` 从真实图片生成完整CSV并做ZIP离线检查。`combine_models.py` 检查不同架构的粗粒度概率平均比例，严格对齐样本ID并计算总参数，未进行额外梯度更新。`finish_trials.py` 和 `finalize_generalize.py` 是本次固定日期的一次性编排/交付工具，不要重复覆盖已完成目录。完整探索结果见 `runs/继续优化实验_20261025.md`。

本机使用 Python3.10、torch2.9.1+cu128 和 torchvision0.24.1+cu128；依赖快照为 `environment/requirements-v2-lock.txt`。裁剪需要 ultralytics8.3.228，安装时保留本项目现有 PyTorch 版本。官方 Linux 环境尚未实际验证。

`inference.py` 的赛事接口只加载本地权重，不在线下载。离线检查拦截 Python 网络连接，不能替代组织方隔离环境测评。公开验证集已用于模型选择，分数不能当作隐藏测试成绩。
