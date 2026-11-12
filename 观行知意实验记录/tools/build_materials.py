"""一次性整理项目文档与官方工作副本；不修改competition/official中的原版。"""
from pathlib import Path
import hashlib
import json
import sys
import textwrap

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

R = Path(__file__).resolve().parents[1]


def save(name, content):
    p = R / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).strip() + '\n', encoding='utf-8')


def js(name, value):
    save(name, json.dumps(value, indent=2, ensure_ascii=False))


save('README.md', r'''
# 观行知意比赛工作区

这是一个面向“学过CS50P、刚接触深度学习”的可运行起点。目标是把一段同步深度图和IR图像分类为40种日常动作之一。先理解数据和流程，再开展完整实验。

## 先做这三步

在VS Code选择“文件 → 打开文件夹”，打开本目录。终端使用PowerShell，工作目录必须是项目根目录。

```powershell
# 1. 检查GPU环境，实际进行张量计算和反向传播。
.\.venv\Scripts\python.exe tools/project.py check-env
# 2. 查看你们的数据。生成runs/preview.png，点击图片即可查看。
.\.venv\Scripts\python.exe tools/project.py preview
# 3. 读取已完成的检查记录，再学习代码；不要立即重复长任务。
Get-Content runs/验收记录.md -Encoding UTF8
```

VS Code右下角选择本项目`.venv\Scripts\python.exe`。按Ctrl+Shift+B可运行环境检查；按F5可调试数据预览。命令使用明确的解释器路径，不依赖是否激活环境。

如果换电脑或环境丢失，执行`powershell -ExecutionPolicy Bypass -File environment/setup_windows.ps1`，详见环境说明。

## 阅读顺序

1. [赛题与交付](docs/01_赛题与交付.md)：我们要做什么、评分和限制。
2. [物品与数据用途](docs/02_物品与数据用途.md)：每个目录、清单、图像、缓存、权重的用途。
3. [学习路径](docs/03_学习路径.md)：按实际成果学习，不用先学完所有数学。
4. [代码导读](docs/04_代码导读.md)：从输入到输出，读懂关键张量和训练步骤。
5. [操作手册](docs/05_操作手册.md)：小规模检查、完整训练、预测、分析与打包。
6. [AI协作](docs/06_AI协作.md)：复制提示词，让AI解释、排错和协助实验。
7. [AutoDL复现](docs/07_AutoDL复现.md)：与官方环境对应的Linux操作。

## 重要的区分

- `competition/official`是原始官方代码，`baseline`是加了注释和少量兼容修正的工作副本。
- `data/raw`为原始训练/验证资料；不移动、不重标、不将验证集用于梯度更新。
- `weights/pretrained.pth`是通用公开预训练参数；`runs/*/*.pt`才是本题训练产生的模型。
- `runs/smoke`和`submission/demo`仅证明工程流程能运行，不能代表模型效果，也不能作为正式成果提交。
- 本地使用适配显卡的CUDA12.8版PyTorch。官方参考环境为CUDA11.8，两套环境不能混为一谈；Linux是否验证见验收记录。

正式提交时官网需要代码ZIP、466段预测CSV、技术报告PDF、网页说明；权重另发赛事管理QQ。此项目不会自动替你们提交或发送文件。
''')

save('docs/01_赛题与交付.md', '''
# 赛题与交付

模型接收一个人的短动作片段，返回按固定顺序排列的40个分数；取最大分数对应的类别。喝水、洗脸和扫地都是动作类别。结果不是动作描述文本。

## 我们需要完成的工作

读取比赛图片 → 同步取帧 → 缩放和通道合并 → 微调视频模型 → 用公开验证集选择方案 → 保存最终权重 → 生成预测和报告 → 演示与答辩。

## 数据与开发边界

训练1995段、验证466段；组织方测试470段不对选手开放。每段最多32对304×228图片，深度为伪彩RGB，IR为8位灰度。三个集合按人物划分。类别编号0到39必须完全沿用manifest.classes的数组位置。

只用本届提供的深度图和IR进行训练/监督/蒸馏/推理。验证集仅做评估和方案选择；不得加入梯度更新。不得使用范围外CUHK片段或组织方测试片段生成伪标签、训练或选模型。通用公开预训练可用，外部CUHK-X预训练/微调权重禁用。参与推理的所有模型参数合计≤100M，冻结参数与检测器也计入。

## 正式交付清单

| 材料 | 去向 | 核对内容 |
|---|---|---|
| 训练/推理代码ZIP | 官网 | 根目录inference.py、config.json；≤20MiB、≤256文件、解压≤64MiB；不带大权重 |
| 公开验证预测CSV | 官网 | 完整466段；sample_id,predicted_class；编号0—39；标识来自清单 |
| 技术报告PDF | 官网 | 中文正文3000—5000字，按模板，引用来源、实验和团队分工 |
| 网页作品说明 | 官网 | 结构、依赖、入口、复现方法；不能代替PDF |
| 最终模型权重 | 赛事管理QQ | 注明队名，与代码/配置一致；文件名等于weights_file |

额外依赖需提前向组织方登记；测评中不联网安装或下载。正式推理离线，不调用在线大模型API。代码或模型更新后官网与QQ权重必须同步更新。

## 评分与时间

自动表现50%、技术路线说明10%、答辩40%。自动原始分：基准60、提升20、40类Macro F1占10、效率10。验证集成绩不是正式比赛成绩。详细公式见官方开发包scoring-rules.pdf，参考值由组织方固定。

单次运行最多1800秒，包括准备、初始化和推理；排队不计。第一段预测计入准确率，但不计入平均推理耗时。每队有100元AutoDL券，先做短实验估算成本。早期PDF的阶段比例已被本次详细规则取代；早期11月20日/30日为暂定日期，应以组织方后续通知为准。
''')

save('docs/02_物品与数据用途.md', '''
# 物品与数据用途

| 物品 | 用途 | 是否提交 |
|---|---|---|
| competition | 早期PDF、当前规则、原版开发包和来源 | 作为参考，不整包提交 |
| data/raw/train | 已知标签的动作片段，用于更新参数 | 不提交 |
| data/raw/validation | 未见人物的已知标签片段，用于评估和选模型 | 不提交原图 |
| manifest.json | 定位片段、标签、40类顺序、内容摘要 | 程序读取，不随意修改 |
| provenance.json | 来源、人物和样本标识，核对划分 | 只供准备与审计，不输入模型 |
| Depth_Color | 伪彩距离图，颜色不是物体真实颜色 | 不单独提交 |
| IR | 同一时刻的红外灰度图 | 不单独提交 |
| data/classes.csv | 类别编号、中英文解释、两集合数量 | 阅读与实验用 |
| baseline | 可运行代码，加中文注释的官方工作副本 | 打包工具挑选所需代码 |
| environment | 环境安装、自检、训练配置、依赖锁定 | 提交包附必要说明 |
| weights/pretrained.pth | 通用预训练知识，用来初始化模型 | 无需提交预训练文件 |
| cache/*.npy | 把图片提前变为定长数组，加快训练 | 不提交，可重新生成 |
| runs | 模型、配置、日志、CSV、分析报告 | 最终版选择后提交相应文件 |
| reports | 原版模板、写作指导、实验记录 | 最终提交PDF |
| submission | 检查后的代码包、单独模型、检查记录 | 按官网/QQ分别提交 |
| .venv、environment/python | 解释器与安装库，仅属于这台电脑 | 不复制到代码包 |

## 一个样本怎样变成模型输入

用train/manifest.json中一条记录的file找到片段目录；Depth_Color和IR按原始帧号对应。均匀选择16个时间位置；不足16帧时重复选择已有帧，不凭空生成动作。两种图采用相同采样顺序和几何变换。

每张深度图变为3通道，IR为1通道，缩放到128×128，除255变为0—1浮点数。组合成[C,T,H,W]=[4,16,128,128]；增加batch维后为[B,4,16,128,128]。模型内部再用固定均值/标准差归一化。输出[B,40]。

伪彩深度图不能当作原始毫米深度计算真实距离。文件名不作为模型特征。标签和人物来源仅用于监督/核验，不能塞进输入。

缓存16帧四通道uint8的完整训练+验证数组约2.4GiB；float32训练时才转换。数据复制件约4.54GiB。预训练和每个检查点各约数百MiB，注意不要把所有实验目录复制进提交包。
''')

save('docs/03_学习路径.md', r'''
# 从CS50P到第一个动作识别实验

CS50P已经提供函数、异常、文件、模块等基础。现在重点是数组和“程序怎样从数据学参数”，不用重学整门Python。

## 第一阶段 看懂数据

学习Path、JSON字典/列表、NumPy数组、图像的宽高和通道。打开一条manifest记录，找到对应目录，运行preview。解释RGB与L模式；指出同一帧的深度与IR。

练习：在VS Code调试tools/project.py的preview，在row变量停下，查看file、label及classes[label]。完成标准：你能自行定位“喝水”样本，并确认类别编号6来自清单，而不是猜测。

## 第二阶段 理解张量与训练

张量是有形状的多维数组；参数是模型需要学习的数。前向传播产生40个分数；交叉熵衡量真实类别分数是否足够高；反向传播计算梯度；优化器利用梯度调整参数。

练习：读smoke函数并观察x、y、logits的shape。解释zero_grad、backward、step的顺序。完成标准：说清batch维、时间维、40类输出，以及训练时为何需要标签、推理时为何不需要。

## 第三阶段 跑通并重新加载

学习Dataset提供单个样本、DataLoader组成batch、model.train()/eval()的差异、Dropout、BatchNorm、state_dict、with no_grad/inference_mode。先读已有smoke日志；需要重跑时先重命名旧runs/smoke保留记录。

完成标准：保存一个检查点，新进程重新加载后输出40个有限值。知道只跑几步不可能证明比赛效果。

## 第四阶段 理解视频与迁移学习

先理解2D卷积提取空间特征，再理解时间上的卷积。R(2+1)D将空间与时间卷积分解；34指网络层级。公开IG65M/Kinetics模型原有3通道，工作代码保留原3通道参数，并用平均值初始化新增IR通道；最后分类头替换成40类。

练习：从RawModel.__init__追到forward；写出[1,4,16,128,128]到[1,40]的数据流。讲清预训练、微调、最终权重的关系。完成标准：解释只看一帧为何容易混淆“举杯”和“喝水”。

## 第五阶段 实验和指标

学习准确率=全部预测正确的比例；每类F1综合精确率和召回率；Macro F1按固定40类平均，缺失类按0处理。混淆矩阵行是真实类，列是预测类。分清验证方案选择和测试正式评分。

先跑一次完整16帧基线，再只改变一个变量：取帧或学习率；记录运行名、种子、配置、权重、准确率、F1、耗时、显存。32帧实验需要同步重建缓存及推理配置，不能只改config.json。

完成标准：用证据解释改动是否有效；不能只说“模型更先进”。不反复追逐验证集分数而隐瞒尝试次数。

## 第六阶段 复现、报告和答辩

从新终端按命令生成完整CSV并打包；在Linux参考环境复现。以reports/报告写作指导.md为骨架填写真实结果。每位队员都能讲输入、模型、实验、错误样本和限制。

参考资料按需阅读：
- Python pathlib：https://docs.python.org/3.10/library/pathlib.html
- NumPy入门：https://numpy.org/doc/stable/user/absolute_beginners.html
- PyTorch基础：https://pytorch.org/tutorials/beginner/basics/intro.html
- 视频模型实现：https://github.com/moabitcoin/ig65m-pytorch

建议按阶段成果安排学习时间，而不是预设“几天精通”。有不懂的地方，把本地代码、变量形状和运行结果交给AI解释，再亲手验证。
''')

save('docs/04_代码导读.md', '''
# 代码导读与调试位置

## 先读操作入口

tools/project.py集中处理路径和运行顺序。先读ROOT、read、write，再读preview、environment、smoke，最后看prepare、train、predict、package。命令行--help列出入口。

## 核心模块

| 模块 | 要读懂的内容 | 推荐观察点 |
|---|---|---|
| baseline/depth_ir_io.py | 帧号排序、同步、均匀采样、resize、/255、C/T/H/W排列 | read_frame_lists返回的shape |
| baseline/raw_model.py | 3→4通道扩展、预训练加载、40类头、标准化 | self.mean、net.stem[0]、head |
| baseline/ig65m_backbone.py | R(2+1)D34架构与权重兼容细节 | 架构学习进阶再读 |
| baseline/training/prepare_training_manifest.py | 合并清单但保留split，人物不交叉 | subject与split；合并不是合并训练 |
| baseline/training/build_cache.py | 把图片存成uint8数组，保存采样规格 | shape_per_sample、counts |
| baseline/training/train_raw.py | 随机增强、AdamW、混合精度、EMA、验证 | loss、model.train、evaluate |
| baseline/inference.py | 只加载本地最终权重，返回40分数 | Model.predict；与训练预处理一致 |
| baseline/predict.py | 官方CSV生成方式和sample_id格式 | file后加.npy是数据协议，不是图像文件 |
| baseline/analyze_predictions.py | 固定40类指标和错例 | confusion_matrix、per_class |

## 容易误解的几处

1. 缓存把train与validation写进同一元数据文件，但Dataset按split筛选，验证数据不参与梯度更新。
2. 训练时随机裁剪/翻转要对整段、所有通道同步执行，否则动作和双模态会错位。
3. eval不等于关闭梯度；推理使用eval加inference_mode。训练loss下降也不等于未见人物上表现更好。
4. 官方--smoke只做前向检查。本项目smoke额外做真实参数更新，checkpoint不是EMA权重。
5. 正式训练官方脚本同时保存best_ema与last_ema；项目入口选择验证准确率最高的best_ema，见selection.json。官方summary中的last指标不能误写成best指标。
6. 不要修改IG65M架构尺寸或BN epsilon，否则公开权重strict加载会失败。

修改记录见competition/工作副本修改记录.json；原版始终保留。training目录存在若干模块副本，修改共享读取或模型代码时必须同步两个位置，并运行回归检查。
''')

save('docs/05_操作手册.md', r'''
# 开发操作手册

下面所有命令在项目根目录PowerShell执行。使用Windows路径，Linux对应说明见07文档。已有产物先保留，不要覆盖实验。安装后的版本和验收状态见runs。

## 数据与预览

```powershell
.\.venv\Scripts\python.exe tools/project.py check-data
.\.venv\Scripts\python.exe tools/project.py preview --split validation --index 0
.\.venv\Scripts\python.exe -m unittest discover -s tools -p "test_*.py"
```

check-data逐帧解码、验证官方SHA256、类别/数量/帧同步和人物隔离，可能需要数分钟。不要为练习反复复制4.54GiB图片；修改代码不会改变原图。

## 小规模流程

```powershell
.\.venv\Scripts\python.exe baseline/download_pretrained.py
.\.venv\Scripts\python.exe tools/project.py prepare --small
.\.venv\Scripts\python.exe tools/project.py smoke --steps 3
.\.venv\Scripts\python.exe tools/project.py predict --run runs/smoke --limit 4
.\.venv\Scripts\python.exe tools/project.py package --run runs/smoke --demo
```

第一次交付时已运行的步骤以验收记录为准。再次smoke前手动把runs/smoke重命名保存；程序拒绝覆盖已有模型。smoke只选4段训练、4段验证的不同动作，用于检查流程；模型分数没有研究意义。不要提交demo-code.zip或smoke.pt。

## 完整基线训练

```powershell
# 这两步会产生完整缓存并进行较长训练；本次交付不自动执行。
.\.venv\Scripts\python.exe tools/project.py prepare
.\.venv\Scripts\python.exe tools/project.py train --name full16
.\.venv\Scripts\python.exe tools/project.py predict --run runs/full16
.\.venv\Scripts\python.exe tools/project.py package --run runs/full16 --name final
```

完整训练配置在environment/train-config.json：16帧、128×128、4通道、无人物裁剪、batch2、workers0、30轮、lr5e-5、seed42。适配本机12GiB显存的保守起点，实测后再调整；内存不足先降batch为1，不要混改输入规格。新实验更换--name，保存配置记录。验证指标用于选择best_ema，不是组织方测试成绩。

若公开权重下载失败，不要用随机权重冒充预训练；检查网络后重新下载，官方脚本会验证SHA256。构建缓存中途失败时保留failures.json定位；重命名不完整目录后再生成，不能只伪造manifest让程序继续。

输出CSV使用官方sample_id（通常是散列目录名加.npy）。正式必须466条，列名严格为sample_id,predicted_class。HTML分析可在浏览器打开，JSON包含40×40混淆矩阵与每类F1。

## 提交准备

1. 确定最终运行目录及selection.json，记录checkpoint SHA256。
2. 生成完整验证CSV，执行正式package。工具选取代码文件，不把整个项目压缩。
3. 核对submission/final中的code.zip、validation.csv、model.pt及离线检查记录。
4. 完成技术报告PDF和网页说明。代码包通过检查不代表所有参赛材料完成。
5. 在AutoDL/Linux参考环境复测；额外依赖提前登记。然后官网提交代码/CSV/PDF/网页说明，QQ另发同一model.pt并注明队名。

本地socket连接拦截检查证明当前Python推理路径没有请求网络；它不是系统级网络隔离，不能替代组织方Linux测评。
''')

save('docs/06_AI协作.md', '''
# 怎样借助AI而且真正学会

AI适合读代码、解释概念、建议实验、分析报错和整理表达。训练和正式预测由本地PyTorch模型完成。比赛推理不能调用在线大模型。向AI提供代码和日志即可；通常不用上传整套数据或个人账号信息。

## 读代码与张量

提供：一个函数、调用位置、实际shape输出，以及“只学过CS50P”的背景。

提示词：
> 我学过CS50P，没学过深度学习。请解释这段代码的输入、输出和每个张量维度，使用本项目[1,4,16,128,128]到[1,40]的例子。先用普通语言讲流程，再解释每行；给我三个检查理解的小问题。不要修改模型架构。

验证：在断点查看实际shape，自己回答问题，再向队友解释一次。AI的形状推断必须与实测一致。

## 定位报错

提供：完整traceback、执行命令、当前目录、Python/Torch/CUDA版本、最近一次修改；不要只截最后一行。

提示词：
> 以下是命令、版本和完整报错。请先区分路径、依赖、数据、显存或逻辑问题，指出证据，给出最小修复。不要整体重装环境，不要删数据，不要改类顺序。说明修复后应运行哪个最短验证命令。

验证：只应用一个修复，重跑触发报错的最小命令，记录结果。AI说“应该能跑”不算通过。

## 设计实验

提供：基线配置、验证准确率/Macro F1、混淆矩阵、耗时、显存、已有尝试和预算。

提示词：
> 请根据这些真实结果提出下一次最小实验，只改变一个变量。说明假设、控制变量、需保存的文件、成功与失败怎样解释。不能使用验证集更新参数，不能引入额外CUHK数据；总推理参数≤100M。先估算时间和显存，再考虑多模型。

验证：建立新运行名和配置，真实训练并比较。不能把AI建议的“预期提升”写进实验结果。

## 核对材料

提供：ZIP文件清单、config、权重SHA256、CSV摘要、报告提纲；通常不必上传大模型。

提示词：
> 按docs/01_赛题与交付.md逐项核查材料，指出证据和缺项。重点检查Model接口、40类顺序、466条CSV、weights_file对应关系、离线运行、体积、引用和团队贡献。未实际运行的内容请标记未验证，不能推定通过。

验证：执行package与Linux复现，人工阅读最终PDF。通过清单不等于组织方评分。

## 写报告与答辩

提供：真实配置、日志、图表、引用来源和团队实际分工。

提示词：
> 请将这些事实整理成模板所需的方案与实验分析，保留局限和未完成项，不补造结果、引用或分工。然后作为评委逐个提问：为什么双模态、为何16帧、预训练来源、如何防止数据泄漏、易混淆动作、速度与精度的取舍。等我回答后再追问。

验证：每个数字都能追到运行记录，每个队员能解释核心代码。把AI协助用于何处写入工具使用说明，并依赛事实际要求披露。
''')

save('docs/07_AutoDL复现.md', r'''
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
''')

save('docs/08_提交包运行说明.md', '''
# 观行知意代码包运行说明

本包为深度图+IR四通道R(2+1)D34动作分类实现。config.json与inference.py位于根目录；训练代码在training。类别顺序见classes.json，与公开manifest一致。model.pt通过赛事管理QQ另发，不在代码ZIP中。

参考环境：Python3.10、torch2.5.1+cu118、torchvision0.20.1+cu118；其他基础依赖见requirements.txt。Windows开发使用torch2.9.1+cu128/torchvision0.24.1+cu128，跨版本兼容状态以团队附带记录为准。测评前完成依赖准备，推理过程中禁止临时安装或下载。

运行入口负责读取config.json，将同目录model.pt的绝对路径写入config["weights_path"]，将包目录写入config["package_dir"]，然后实例化Model(config, device)。predict(sample_path)读取Depth_Color与IR，同步均匀取16帧、128×128，返回40个有限分数。推理不需要训练清单、验证标签、缓存或预训练文件。

本地验证：python predict.py --dataset /path/to/validation --config config.json --output validation.csv。公开样本标识遵循manifest；当前格式为file字段加.npy，不要擅自去掉后缀。

训练：先用training/prepare_training_manifest.py生成保留split的清单，training/build_cache.py加--include-ir制作16帧128尺寸缓存，再用training/train_raw.py微调。--help可查看参数。公开预训练来源、SHA256和训练配置应随技术报告提交。默认person_crop=false，不使用额外检测器。
''')

save('reports/报告写作指导.md', '''
# 技术报告写作指导

保留原版docx模板的版式，最终导出PDF。正文中文3000—5000字，封面、目录、参考文献、附录不计；中文摘要不超过300字。原模板有中文和英文摘要、关键词、五章正文及参考文献。模板中的占位符不是实验结论。

| 位置 | 建议填写 | 证据来源 |
|---|---|---|
| 封面 | 赛道A、观行知意、作品名、队名、成员、提交日期 | 实际团队信息 |
| 摘要/Abstract | 任务、方法、真实结果、主要结论；未训练时结果留空 | 最终实验 |
| 第一章 任务理解与数据 | 40类、双模态、人物隔离、限制、指标 | 本次详细规则、数据校验 |
| 第二章 方法与系统设计 | 流程图、16帧、4通道、R34、预训练、分类头、推理接口 | 配置与模型代码 |
| 第三章 工程实现与复现 | 环境、缓存、训练/验证分离、权重和入口、相对路径、离线 | 环境检查、操作手册、提交包 |
| 第四章 实验与分析 | 完整基线、单变量对比、准确率/F1、混淆矩阵、错例、耗时 | runs真实日志及CSV |
| 第五章 总结与展望 | 有证据的收益、局限、未完成改进、团队贡献、AI实际用途 | 实际工作记录 |
| 参考文献 | 论文、开源代码、数据、工具，按GB/T7714—2015 | 确认过的原始链接 |

可按正文约600、900、700、1400、500字分配五章，再调整总字数。写作重点是解释选择和实验，不要堆砌网络名词。

至少准备数据流图、完整基线指标表、一个单变量对比表、混淆矩阵/类别表现和几个错例。smoke的4段验证不是完整基线结果，不能拿来证明模型性能。

公开预训练记录必须含模型名称、来源、许可证、SHA256、训练数据、微调设置；见weights/pretrained-source.json。来源仓库注明MIT，权重来自IG65M并微调Kinetics400，报告应准确区分代码许可证与数据/权重使用条件，保留实际查阅链接，不自行编造额外授权。

待填写：完整验证准确率【待运行】；Macro F1【待运行】；推理延迟【待运行】；Linux复现【待验证】；实验对比【待完成】；各成员贡献【团队填写】。最终PDF需检查中文字体、图表标注、目录页码、参考文献和所有占位符。
''')
save('reports/实验记录.csv', 'run_name,日期,负责人,模型,预训练SHA256,帧数,尺寸,通道,batch,lr,seed,epochs,改变变量,accuracy,macro_f1,latency_ms,peak_gpu_mib,权重SHA256,结论\n')
save('reports/答辩提纲.md', '''
# 答辩提纲

按实际答辩时长安排：任务与数据 → 输入和模型 → 工程演示 → 完整实验与错例 → 局限与改进 → 分工。

准备回答：为何不是单帧分类？IR新增了什么信息？为何16帧？如何处理短片段？预训练来自哪里？如何保证人物不泄漏？40类顺序怎样保持？验证集用于什么？为什么同时看准确率和Macro F1？延迟包含哪些步骤？权重与代码怎样对应？

演示用同一版本代码和权重，准备一段正常例子与一个错误例子。说明错误原因的证据和不确定性，不能仅展示成功例子。技术亮点应来自已完成实验。
''')

save('environment/requirements-base.txt', '''
numpy==2.2.6
Pillow==12.2.0
timm==1.0.27
''')
save('environment/requirements-windows.txt', '''
# 先按setup_windows.ps1从官方cu128索引安装torch和torchvision，再安装requirements-base.txt。
torch==2.9.1+cu128
torchvision==0.24.1+cu128
numpy==2.2.6
Pillow==12.2.0
timm==1.0.27
''')
save('environment/setup_windows.ps1', r'''
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw '需要uv管理独立Python。安装说明：https://docs.astral.sh/uv/getting-started/installation/'
}
$installDir = Join-Path $PSScriptRoot 'python'
uv python install 3.10 --install-dir $installDir
if ($LASTEXITCODE -ne 0) { throw 'Python下载失败' }
$pythonPath = (Get-ChildItem -LiteralPath $installDir -Directory -Filter 'cpython-3.10.*' | Sort-Object Name -Descending | Select-Object -First 1).FullName
$pythonPath = Join-Path $pythonPath 'python.exe'
$venvDir = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    uv venv --python $pythonPath $venvDir
    if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败' }
}
uv pip install --python $venvPython torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch安装失败' }
uv pip install --python $venvPython -r (Join-Path $PSScriptRoot 'requirements-base.txt')
if ($LASTEXITCODE -ne 0) { throw '基础依赖安装失败' }
uv pip freeze --python $venvPython | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'requirements-lock.txt') -Encoding utf8
& $venvPython (Join-Path $projectRoot 'tools/project.py') check-env
if ($LASTEXITCODE -ne 0) { throw 'GPU自检未通过' }
Write-Host '安装完成。VS Code请选择本项目.venv/Scripts/python.exe。'
''')
js('environment/train-config.json', {'batch': 2, 'epochs': 30, 'workers': 0, 'lr': 5e-5, 'seed': 42, 'device': 'cuda'})
save('environment/README.md', '''
# 环境说明

本机Python3.10安装在本目录python下，项目依赖在../.venv。它们不影响旧项目Python3.14的CPU环境。VS Code已安装Python和调试扩展。

setup_windows.ps1可以重建环境；requirements-base为官方基础依赖，requirements-windows为本机关键版本，requirements-lock为实际安装的全部依赖。PyTorch从cu128索引单独安装，其他库再安装。无需为本模型单独安装CUDA Toolkit或编译器，实际GPU自检是判断依据。

official-reference.json保存原版环境信息。不要在Windows直接执行官方setup.sh覆盖显卡适配版本。完整训练参数在train-config.json；输入规格还由缓存构建命令决定，修改帧数/大小需要同步缓存和推理配置。
''')

js('.vscode/settings.json', {'python.defaultInterpreterPath': '${workspaceFolder}/.venv/Scripts/python.exe',
    'python.terminal.activateEnvironment': True, 'files.encoding': 'utf8',
    'files.exclude': {'**/__pycache__': True}, 'search.exclude': {'data/raw/**': True, 'cache/**': True, 'weights/**': True, 'environment/python/**': True, '.venv/**': True}})
tasks = []
for label, args in [('环境检查', ['check-env']), ('数据校验（全部图片）', ['check-data']), ('同步图片预览', ['preview']),
                    ('准备小缓存', ['prepare', '--small']), ('小规模训练（已有结果会拒绝覆盖）', ['smoke']),
                    ('小规模预测', ['predict', '--run', 'runs/smoke', '--limit', '4']),
                    ('完整基线训练（耗时较长）', ['train']), ('完整验证预测', ['predict'])]:
    task = {'label': label, 'type': 'process', 'command': '${workspaceFolder}/.venv/Scripts/python.exe',
            'args': ['${workspaceFolder}/tools/project.py', *args], 'options': {'cwd': '${workspaceFolder}'}, 'problemMatcher': []}
    if label == '环境检查':
        task['group'] = {'kind': 'build', 'isDefault': True}
    tasks.append(task)
js('.vscode/tasks.json', {'version': '2.0.0', 'tasks': tasks})
js('.vscode/launch.json', {'version': '0.2.0', 'configurations': [
    {'name': '调试数据预览', 'type': 'debugpy', 'request': 'launch', 'program': '${workspaceFolder}/tools/project.py',
     'args': ['preview'], 'console': 'integratedTerminal', 'cwd': '${workspaceFolder}', 'justMyCode': True},
    {'name': '调试GPU自检', 'type': 'debugpy', 'request': 'launch', 'program': '${workspaceFolder}/tools/project.py',
     'args': ['check-env'], 'console': 'integratedTerminal', 'cwd': '${workspaceFolder}', 'justMyCode': True}]})
js('.vscode/extensions.json', {'recommendations': ['ms-python.python', 'ms-python.debugpy']})
js('观行知意.code-workspace', {'folders': [{'path': '.'}], 'settings': {}})
save('.gitignore', '''
.venv/
environment/python/
__pycache__/
*.pyc
data/raw/
cache/
weights/*.pth
runs/**/*.pt
runs/**/*.npz
submission/**/model.pt
competition/official-devkit.zip
''')

save('competition/当前详细规则.md', '''
# 当前规则备份

来源：用户本次提供的赛题详细说明。此文件是条款整理，并非官网页面的逐字快照。早期PDF仅作历史参考。官方开发包来源、哈希见sources.json。

任务：40类短片段动作分类；CUHK-X的伪彩深度Depth_Color与灰度IR，最多32对同步304×228帧。训练1995、验证466、封闭测试470，按人物隔离。类别0—39与classes数组位置完全一致。

开发：推荐R(2+1)D34，示例16帧、128×128、4通道。数据处理/训练在本地或租用GPU上完成，每队100元券。公开验证集评估与选择，不用于参数学习。允许深度/IR单模态或融合，允许调整取帧、尺寸、裁剪。禁止额外模态训练、辅助监督、蒸馏或推理；禁止本届范围外CUHK片段额外训练。禁止组织方测试数据参与训练、伪标签或选择。

权重：可用通用公开预训练，禁止外部CUHK-X预训练或微调权重。报告列模型、来源、许可证、SHA256、训练数据、微调设置。全部推理参数（含检测器、分类器、冻结参数）≤100M。

接口：ZIP根目录inference.py、config.json；Model(config,device)，predict(sample_path)读取片段，返回按类别顺序的40个有限数值，不要求和为1。示例运行入口将weights_file解析为weights_path，禁止写死自己电脑的绝对路径。CSV严格sample_id,predicted_class，类别0—39。

离线：仅运行训练好的模型，不调用在线大模型API。依赖采用配套环境；额外依赖提前登记，测评中不临时联网安装或下载。

官网：代码/配置ZIP（≤20MiB，≤256文件，解压≤64MiB，不放大权重）、最终466段公开验证CSV、技术报告PDF、网页作品说明。网页说明涵盖结构、入口、依赖和复现，不代替PDF。最终权重直接发赛事管理QQ并注明队名；文件名必须对应weights_file。其他参与推理权重一并发送并注明用途。变更后同步更新官网与QQ材料，按最终版本测评。

评分：自动表现50%、技术说明10%、现场答辩40%。自动原始分=基准60+提升20+40类Macro F1 10+效率10；验证成绩不直接计入正式成绩。详细公式和参考见scoring-rules.pdf，参考值在开放测评前固定。1800秒上限包含领取后程序包/权重准备、初始化、推理，排队除外。首样本用于预热，预测计入准确率，耗时不计平均延迟。

报告：中文正文3000—5000字，不含封面、目录、文献、附录；采用官方模板，交PDF，含摘要、正文、总结、参考。鼓励架构/流程图、实验图表、环境截图。引用论文、代码、工具需说明来源与实际用途。评审看方案、实现、分析、表达与团队贡献。

答辩：方案、技术亮点、演示、问答，时间地点形式另行通知。最终成绩=自动原始分×50%+技术实际分0—10+答辩实际分0—40。

配套开发教程：https://uestcai.top/news/community-00d3bcac-0350-43f4-b91c-d2173e99d592
开发包：https://uestcai.top/api/markdown-assets/cf37e5ba-69c8-4f3e-b9ae-db59c518f61c
评分PDF：https://uestcai.top/api/markdown-assets/c5f5f808-2844-479e-9121-a537f2b7be49
报告模板：https://uestcai.top/api/markdown-assets/c727779d-75d8-4ffe-9c3c-0636c8bdf7e8
数据分享：https://pan.baidu.com/s/1AW7joSver7Qfu681xAOg8w?pwd=1pgi
''')

for folder, explanation in {
    'data': 'raw是从用户给定解压目录复制的原始数据；classes.csv由数据检查生成，不要按类别重排原图。',
    'cache': '清单和uint8数组由prepare生成；small16为流程检查，full16为完整训练。缓存可重建，切换平台需重建绝对路径清单。',
    'weights': 'pretrained.pth仅用于训练初始化，来自官方脚本指定IG65M/Kinetics公开权重；最终权重在runs内。',
    'runs': '按运行名保存实验。smoke只有少量训练步骤；data-check为完整数据校验；完整实验需单独启动。',
    'submission': 'demo是工程演示包，不可提交。正式包需用完整训练权重和466条CSV生成；报告PDF和网页说明另行准备。',
    'competition': 'official-devkit.zip与official目录保留原版；baseline工作副本单独修改。当前规则优先于早期PDF。',
    'baseline': '原版来自competition/official；此处补中文注释、同步校验和实际配置说明。用项目tools/project.py从根目录运行。',
    'reports': '保留用户提供的原始Word模板。学习/实验阶段先填Markdown和CSV，最终按模板形成真实技术报告PDF。'
}.items():
    save(f'{folder}/README.md', f'# {folder}用途\n\n{explanation}')

# 从原始文件重新生成工作副本，注释可重复生成，原版的哈希不变。
annotations = {
    'depth_ir_io.py': [('def uniform_indices', '# 均匀取帧：短片段允许重复位置，深度和IR必须采用相同索引。\ndef uniform_indices'),
                       ('            value = np.asarray', '            # uint8的0—255转为float32的0—1，模型内部还会标准化。\n            value = np.asarray'),
                       ('    return np.stack', '    # 单帧[C,H,W]沿时间维堆叠，得到[C,T,H,W]。\n    return np.stack')],
    'raw_model.py': [('        if pretrained:', '        # 先严格加载公开3通道模型，架构必须完全匹配。\n        if pretrained:'),
                     ('        if channels==4:', '        # 原RGB核保留，新IR通道用三个通道的平均核初始化。\n        if channels==4:'),
                     ('        self.head=', '        # 丢弃原400类头，重新学习本比赛40类动作。\n        self.head='),
                     ('    def forward', '    # x为[B,4,T,H,W]；广播归一化，输出[B,40]原始分数。\n    def forward')],
    'inference.py': [('class Model:', '# 测评只调用此接口；初始化期间仅加载本地最终权重，不下载预训练模型。\nclass Model:'),
                     ('        weights =', '        # 运行入口将config中的相对weights_file解析为weights_path。\n        weights ='),
                     ('        return self.model', '        # 40个有限分数；顺序严格对应manifest.classes；不必做softmax。\n        return self.model')],
    'training/train_raw.py': [('class CachedClips', '# 数据缓存同时记录两集合，但Dataset按split隔离，验证集不用于梯度更新。\nclass CachedClips'),
                             ('        if self.training:', '        # 同一段所有时间帧和四个通道同步翻转/裁剪。\n        if self.training:'),
                             ('def metrics', '# 混淆矩阵行=真值、列=预测；Macro F1固定平均40类。\ndef metrics'),
                             ('    model.eval();', '    # eval切换BatchNorm/Dropout；inference_mode关闭梯度跟踪。\n    model.eval();'),
                             ('    if a.smoke:', '    # 官方smoke只前向；本项目tools/project.py另有真实更新检查。\n    if a.smoke:'),
                             ('        model.train();', '        # 每轮先训练；后面的evaluate只评估、不更新参数。\n        model.train();'),
                             ('            scaler.scale(loss).backward();', '            # 混合精度缩放损失，再反向传播和裁剪梯度。\n            scaler.scale(loss).backward();'),
                             ('        if m[\'accuracy\']>best:', '        # 项目正式推理默认选择验证准确率最高的best_ema；完整记录尝试次数。\n        if m[\'accuracy\']>best:')],
    'training/prepare_training_manifest.py': [('    for split in', '    # 合并元数据便于制缓存，但每条记录继续保留原split与subject。\n    for split in')],
    'training/build_cache.py': [('    arrays=', '    # uint8磁盘映射避免一次把完整训练数据读入内存；加载训练时再转浮点。\n    arrays=')]
}

sync_function = '''
def synchronized_paths(depth_paths, ir_paths):
    """按真实帧号核对两种模态；不同数量、重复帧或缺失帧不能静默对齐。"""
    def keys(paths):
        result = []
        for path in paths:
            match = re.search(r'_(\\d+)(?:_Color)?\\.[^.]+$', Path(path).name, re.IGNORECASE)
            key = int(match.group(1)) if match else int(Path(path).stem)
            result.append(key)
        if not result or len(set(result)) != len(result):
            raise ValueError('Empty or duplicate modality frame numbers')
        return result
    if keys(depth_paths) != keys(ir_paths):
        raise ValueError('Depth/IR frames must have identical ordered frame numbers')


'''
for name, changes in annotations.items():
    original = (R / 'competition/official' / name).read_text(encoding='utf-8')
    text = original
    for old, new in changes:
        if old not in text:
            raise RuntimeError(f'注释锚点不存在: {name}: {old}')
        text = text.replace(old, new, 1)
    if name == 'depth_ir_io.py':
        text = text.replace('def read_frame_lists(', sync_function + 'def read_frame_lists(', 1)
        text = text.replace("    depth = _read(depth_paths", "    if include_ir:\n        if not ir_paths:\n            raise ValueError('IR frames are required')\n        synchronized_paths(depth_paths, ir_paths)\n    depth = _read(depth_paths", 1)
    if name == 'training/train_raw.py':
        text = text.replace("'recipe':'YOLO11n on8 IR probes, fixed union1.4 margin/.35 floor; R34 IG65M-K400; dropout .3; AdamW lr5e-5 wd1e-4; warmup2 cosine; EMA.99; same clip flip and crop',", "'recipe':'R34 IG65M-K400; dropout .3; AdamW wd1e-4; warmup2 cosine; EMA.99; synchronous flip/crop',\n          'person_crop':tr.meta['person_crop'], 'input_shape':tr.meta['shape_per_sample'],")
    save('baseline/' + name, text)
    if name in ['depth_ir_io.py', 'raw_model.py']:
        save('baseline/training/' + name, text)
    if name == 'training/build_cache.py':
        save('baseline/build_cache.py', text)

config = json.loads((R / 'competition/official/config.json').read_text(encoding='utf-8'))
config['weights_file'] = '../runs/full16/best_ema.pt'
js('baseline/config.json', config)
(R / 'environment/official-reference.json').write_bytes((R / 'competition/official/environment-reference.json').read_bytes())
files = []
for p in (R / 'competition/official').rglob('*'):
    if p.is_file():
        rel = p.relative_to(R / 'competition/official')
        target = R / 'baseline' / rel
        if target.exists() and p.read_bytes() != target.read_bytes():
            files.append({'file': rel.as_posix(), 'original_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                          'working_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
js('competition/工作副本修改记录.json', {'changes': ['关键代码中文注释', '训练和推理均按帧号严格核对Depth/IR同步',
    '训练元数据记录实际person_crop和输入形状，去除固定YOLO描述', '工作默认配置选择best_ema相对路径'], 'files': files})
js('competition/sources.json', {'official_devkit': {'url': 'https://uestcai.top/api/markdown-assets/cf37e5ba-69c8-4f3e-b9ae-db59c518f61c',
   'sha256': hashlib.sha256((R / 'competition/official-devkit.zip').read_bytes()).hexdigest()},
   'dataset_source': 'D:/桌面文件/cvpr/wujie-classification-depth-ir-data.zip/wujie-classification-depth-ir-data',
   'template_source': 'E:/DOWNLOAD/通用技术报告模板.docx', 'rules_source': '用户本次对话提供的详细规则',
   'early_pdf_source': 'D:/桌面文件/cvpr/观行知意_赛题说明.pdf'})
js('weights/pretrained-source.json', {'model': 'R(2+1)D-34 clip32, IG65M then Kinetics400',
   'source_url': 'https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth',
   'repository': 'https://github.com/moabitcoin/ig65m-pytorch', 'code_license': 'MIT (official backbone reference)',
   'training_data': ['IG65M', 'Kinetics400'], 'cuhk_external_pretraining': False,
   'expected_sha256': 'ade133f18548b59da3a36d6a70d44a4a7cc4b90460654c338cb8a023e9a3e0e1',
   'weight_terms_note': '记录官方配套来源；代码许可证与数据/权重条款需分别查阅，不额外推断授权。',
   'fine_tuning': '本届train，4通道，40类；实际参数见各运行metadata与environment/train-config.json'})
print('文档、中文注释、VS Code配置和环境脚本已生成')
