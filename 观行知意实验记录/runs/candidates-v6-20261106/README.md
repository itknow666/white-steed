# 四种候选模型实验

本目录保存 2026-11-06 的模型筛选计划、预检查、日志和汇总。

- `plan.json`：训练前固定的四模型、学习率、轮数和融合候选。
- `progress.json`：当前队列阶段；训练中不代表实验已完成。
- `results-partial.json`：已经完成的模型；`results.json` 是所有训练结束后的记录。
- `*-probe.json`：真实训练样本的两步前向/反向和权重重载检查。
- `*-probe.pt`：**仅两步检查用权重，不是比赛训练结果，不能当最终模型提交。**
- 每个正式训练结果在相邻的 `../模型名-v6-20261106/`，其中 `best_ema.pt` 是按公开验证集选择的最佳权重，`resume.pt` 用于恢复训练。
- `outcome.json` 和 `实验结果.md`：原始图片复验、代码 ZIP 离线加载通过后生成的最终结果。
- `source-snapshot.zip`：初始源码快照；后续收尾脚本的修改另存最终源码快照。

状态查询（项目根目录）：

```powershell
& ./.venv/Scripts/python.exe optimized/status_candidates.py
```

本轮每个模型仅训练六轮，使用相同训练/验证划分，但根据架构调整批量、学习率和输入网格。这是初筛，不能视为架构的最终能力排名。旧 R34 的预训练来源和本题训练历史也与这些新候选不同。

全部推理仅使用深度和 IR；公共权重仅用于初始化。依赖锁定在 `environment/requirements-lock-v6.txt`；官方来源、SHA256、许可证和适配说明在 `competition/candidates-v6/`。新增依赖需按比赛要求登记；Linux 尚未验证。
