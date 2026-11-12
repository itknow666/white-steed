# 环境说明

本机Python3.10安装在本目录python下，项目依赖在../.venv。它们不影响旧项目Python3.14的CPU环境。VS Code已安装Python和调试扩展。

setup_windows.ps1可以重建环境；requirements-base为官方基础依赖，requirements-windows为本机关键版本，requirements-lock为实际安装的全部依赖。PyTorch从cu128索引单独安装，其他库再安装。无需为本模型单独安装CUDA Toolkit或编译器，实际GPU自检是判断依据。

official-reference.json保存原版环境信息。不要在Windows直接执行官方setup.sh覆盖显卡适配版本。完整训练参数在train-config.json；输入规格还由缓存构建命令决定，修改帧数/大小需要同步缓存和推理配置。
