param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = [System.IO.Path]::GetFullPath($Root)
$RootDrive = [System.IO.Path]::GetPathRoot($Root)
if ($RootDrive -ne "F:\") {
    throw "Safety stop: Root must be located on F:\, current Root=$Root"
}
$Base = Split-Path -Parent $Root
$Runtime = Join-Path $Base "Runtime"
$Uv = Join-Path $Runtime "uv\uv.exe"
$Python = Join-Path $Runtime "envs\dronevehicle-unet\Scripts\python.exe"
$TempDir = Join-Path $Runtime "temp"

$Dirs = @(
    $TempDir,
    (Join-Path $Runtime "cache\uv"),
    (Join-Path $Runtime "cache\pip"),
    (Join-Path $Runtime "cache\torch"),
    (Join-Path $Runtime "cache\cuda")
)
$Dirs | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}

$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:UV_CACHE_DIR = Join-Path $Runtime "cache\uv"
$env:PIP_CACHE_DIR = Join-Path $Runtime "cache\pip"
$env:TORCH_HOME = Join-Path $Runtime "cache\torch"
$env:CUDA_CACHE_PATH = Join-Path $Runtime "cache\cuda"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

if (-not (Test-Path -LiteralPath $Uv -PathType Leaf)) {
    throw "F-drive uv was not found: $Uv"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "F-drive Python was not found: $Python"
}

& $Uv pip install --python $Python numpy==1.26.4
if ($LASTEXITCODE -ne 0) {
    throw "NumPy compatibility installation failed."
}

& $Python -c "import sys,numpy,torch; print('Python=',sys.executable); print('NumPy=',numpy.__version__); print('PyTorch=',torch.__version__); print('CUDA=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if ($LASTEXITCODE -ne 0) {
    throw "Environment verification failed."
}
