param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Prepare", "Preview", "SelfTest", "Train", "Evaluate", "Compare", "All")]
    [string]$Action,

    [Parameter(Mandatory = $true)]
    [string]$Root,

    [ValidateSet("P0", "P1", "P2", "P3", "P4", "P5")]
    [string]$Policy = "P0"
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
$WorkDir = Join-Path $Root "12_model_training"
$Repo = Join-Path $WorkDir "bookish-computing-machine"
$RunDir = Join-Path $WorkDir "dronevehicle_unet_runs\limit_pilot_v1\$Policy"
$Python = Join-Path $Runtime "envs\dronevehicle-unet\Scripts\python.exe"
$Entry = Join-Path $PSScriptRoot "unet_limit_pilot.py"

$TempDir = Join-Path $Runtime "temp"
$CacheDirs = @(
    $TempDir,
    (Join-Path $Runtime "cache\uv"),
    (Join-Path $Runtime "cache\pip"),
    (Join-Path $Runtime "cache\torch"),
    (Join-Path $Runtime "cache\cuda"),
    (Join-Path $Runtime "cache\matplotlib"),
    (Join-Path $Runtime "cache\huggingface")
)
$CacheDirs | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}

$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:UV_CACHE_DIR = Join-Path $Runtime "cache\uv"
$env:PIP_CACHE_DIR = Join-Path $Runtime "cache\pip"
$env:TORCH_HOME = Join-Path $Runtime "cache\torch"
$env:CUDA_CACHE_PATH = Join-Path $Runtime "cache\cuda"
$env:MPLCONFIGDIR = Join-Path $Runtime "cache\matplotlib"
$env:HF_HOME = Join-Path $Runtime "cache\huggingface"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "F-drive Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $Repo "models\unet.py") -PathType Leaf)) {
    throw "Senior U-Net repository was not found: $Repo"
}
if (-not (Test-Path -LiteralPath $Entry -PathType Leaf)) {
    throw "Training entry was not found: $Entry"
}

$ActionMap = @{
    "Prepare" = "prepare"
    "Preview" = "preview"
    "SelfTest" = "self-test"
    "Train" = "train"
    "Evaluate" = "evaluate"
    "Compare" = "compare"
    "All" = "all"
}

Write-Host "Python: $Python"
Write-Host "Repository: $Repo"
Write-Host "Run directory: $RunDir"
Write-Host "Action: $Action"
Write-Host "Policy: $Policy"

& $Python $Entry `
    --action $ActionMap[$Action] `
    --root $Root `
    --repo $Repo `
    --run-dir $RunDir `
    --policy $Policy

if ($LASTEXITCODE -ne 0) {
    throw "Action failed with exit code $LASTEXITCODE"
}
