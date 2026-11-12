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
