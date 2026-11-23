# 方向 A · 跨人物 Mixup 配对

在官方 1,995 段训练片段上，从现有 73.61% 双路权重开始，固定 16 帧、192×192、人物裁剪和原非目标蒸馏。把训练时混合片段的配对方式作为唯一显式变化。Mixup 按同一个系数混合输入并计算两份动作标签损失。本实验的配对人物标识来自官方训练数据的溯源记录。 [Mixup 论文](https://arxiv.org/abs/1710.09412)

| 配对 | 做法 |
|---|---|
| random | 使用原有批内随机排列，允许原片段配给自己 |
| distinct | 用同一次随机排序构造无自配对循环 |
| cross-person | 在同一次随机排序上按人物分组，循环移位尽可能多地配到不同人物 |

不同人物策略在一批中最大化跨人物配对数。若一个人物的片段占过半，仍可能有同人物配对。两种新策略保持一对一映射，每个片段同样用作一次配对来源，不增加视频数据。对齐的深度和 IR 使用同一配对和系数。两个策略可能连带改变不同动作配对的比例，日志单独记录这一项，用于解释观察。

每种策略使用 seed 42 和 43。每个种子三组的批序、Mixup 触发时机、Beta(0.2, 0.2) 混合系数完全相同。各两轮、batch 8、学习率 6e-6。原来的非目标蒸馏缓存只来自官方训练集，软目标沿同一个配对混合，蒸馏温度 2、损失系数 0.35。人物信息仅在训练时用于配对。部署模型仍采用同步深度、IR 与本地人物裁剪。

配对验证穷举每批 1 至 8 条、3 个匿名人物的 9,840 种排列模式，覆盖同人物占过半及单一人物情形，确认跨人物组达到批内理论最大值。包级验证使用 466 段公开原图，并以同种子的随机组计算分人物改对和改错。

## 复现

    $root = 'runs/v8-cross-person-mixup-20261124'
    $py = '.\.venv\Scripts\python.exe'
    $common = @('--mode','dual','--continue-dual','--initial','runs/v8-teacher-transfer-20261122/kd-non-target/best_ema.pt','--teacher-cache','runs/v8-teacher-transfer-20261122/teacher-train.npz','--teacher-policy','non-target','--teacher-weight','0.35','--epochs','2','--base-lr','6e-6','--batch','8','--cache','cache/crop16-192-v3','--audit-mixup')
    foreach ($seed in 42,43) {
        $run = if ($seed -eq 42) { $root } else { "$root/seed43" }
        foreach ($arm in 'random','distinct','cross-person') {
            & $py optimized/train_domain_v8.py @common --seed $seed --mixup-partners $arm --output "$run/$arm"
        }
    }
    & $py optimized/eval_cross_person_mixup_v8.py --runs $root --output submission/v8-cross-person-mixup-20261124

见 [实验结果](../runs/v8-cross-person-mixup-20261124/实验结果.md)。
