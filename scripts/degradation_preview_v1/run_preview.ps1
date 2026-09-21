[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("SelfTest", "Preview", "Verify")]
    [string]$Action,

    [Parameter(Mandatory = $true)]
    [string]$Root,

    [switch]$OpenReports
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ResolvedRoot = [System.IO.Path]::GetFullPath($Root)
$RootDrive = [System.IO.Path]::GetPathRoot($ResolvedRoot)
if ($RootDrive -ne "F:\") {
    throw "Safety stop: Root must be located on F:\, current Root=$ResolvedRoot"
}

$Base = Split-Path -Parent $ResolvedRoot
$Runtime = Join-Path $Base "Runtime"
$Python = Join-Path $Runtime "envs\dronevehicle-unet\Scripts\python.exe"
$ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Program = Join-Path $ToolDir "preview_degradation_limits.py"
$Output = Join-Path $ResolvedRoot "13_degradation_limit_preview_v1"
$TempDir = Join-Path $Runtime "temp"

New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:UV_CACHE_DIR = Join-Path $Runtime "cache\uv"
$env:PIP_CACHE_DIR = Join-Path $Runtime "cache\pip"
$env:TORCH_HOME = Join-Path $Runtime "cache\torch"
$env:CUDA_CACHE_PATH = Join-Path $Runtime "cache\cuda"
$env:MPLCONFIGDIR = Join-Path $Runtime "cache\matplotlib"
$env:HF_HOME = Join-Path $Runtime "cache\huggingface"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "F-drive Python environment was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Program -PathType Leaf)) {
    throw "Preview program was not found: $Program"
}

$PythonAction = if ($Action -eq "SelfTest") { "self-test" } else { $Action.ToLowerInvariant() }
Write-Host "Python: $Python"
Write-Host "Tool: $ToolDir"
Write-Host "Output: $Output"
Write-Host "Action: $Action"

& $Python $Program --action $PythonAction --root $ResolvedRoot
$ActionExitCode = $LASTEXITCODE
if ($ActionExitCode -ne 0) {
    throw "Action failed. Exit code: $ActionExitCode"
}

if ($Action -eq "Preview" -and $OpenReports) {
    $Reports = @(
        (Join-Path $Output "reports\haze_three_levels_preview.png"),
        (Join-Path $Output "reports\dark_three_levels_preview.png"),
        (Join-Path $Output "reports\noise_three_levels_preview.png"),
        (Join-Path $Output "reports\dark_noise_key_combinations_preview.png")
    )
    foreach ($Report in $Reports) {
        if (Test-Path -LiteralPath $Report -PathType Leaf) {
            Start-Process -FilePath $Report
        }
    }
}


