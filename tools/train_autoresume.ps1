#Requires -Version 5.1
<#
.SYNOPSIS
    Restart transformer PPO until train.num_updates is finished.

.DESCRIPTION
    Runs scripts/train_transformer.py with --resume auto. A non-zero exit
    waits RetrySeconds (default 30) and tries again, so a killed python
    process (reboot, sleep, crash) continues from the last complete
    checkpoint. Exit 0 means the target update count was reached and this
    script stops. Exit 2 is a config-hash refusal and is not retried.

    The target is -TargetUpdates when that is greater than 0, otherwise
    train.num_updates in -Config. Paths are resolved from the repo root
    (the parent of this tools directory), not from the scheduler's
    working directory.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,

    [string]$Python = "python",

    [int]$TargetUpdates = 0,

    [string]$LogFile = "",

    [int]$RetrySeconds = 30,

    [switch]$AllowConfigMismatch,

    [string[]]$ExtraArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RepoRoot) {
    throw "Cannot locate the repo root from $PSScriptRoot"
}

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $RepoRoot $PathValue)
}

$ConfigPath = Resolve-RepoPath $Config
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Config not found: $ConfigPath"
}

if ($TargetUpdates -le 0) {
    $payload = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $payload.train -or -not $payload.train.num_updates) {
        throw "Config $ConfigPath has no train.num_updates. Pass -TargetUpdates."
    }
    $TargetUpdates = [int]$payload.train.num_updates
}

if (-not $LogFile) {
    $LogFile = Join-Path $RepoRoot "runs\autoresume.log"
} else {
    $LogFile = Resolve-RepoPath $LogFile
}
$logDir = Split-Path -Parent $LogFile
if ($logDir -and -not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$src = Join-Path $RepoRoot "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$src;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $src
}

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"), $Message
    Add-Content -LiteralPath $LogFile -Value $line -Encoding utf8
    Write-Host $line
}

function Quote-ProcessArg([string]$Value) {
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

$trainScript = Join-Path $RepoRoot "scripts\train_transformer.py"
$argList = @(
    $trainScript,
    "--config", $ConfigPath,
    "--resume", "auto",
    "--updates", "$TargetUpdates"
)
if ($AllowConfigMismatch) {
    $argList += "--allow-config-mismatch"
}
if ($ExtraArgs) {
    $argList += $ExtraArgs
}

$argumentString = ($argList | ForEach-Object { Quote-ProcessArg $_ }) -join " "
Write-Log "autoresume start target_updates=$TargetUpdates config=$ConfigPath"
Write-Log "command: $Python $argumentString"

while ($true) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdoutFile = Join-Path $logDir "train-$stamp.out.txt"
    $stderrFile = Join-Path $logDir "train-$stamp.err.txt"
    Write-Log "launching training"

    # One argument string so Windows PowerShell 5.1 does not re-split paths.
    # WindowStyle exists on Windows only. Redirects already detach the stdio
    # streams; on Windows the hidden style avoids a console flash.
    $startArgs = @{
        FilePath               = $Python
        ArgumentList           = $argumentString
        WorkingDirectory       = $RepoRoot
        Wait                   = $true
        PassThru               = $true
        RedirectStandardOutput = $stdoutFile
        RedirectStandardError  = $stderrFile
    }
    if ($env:OS -eq "Windows_NT") {
        $startArgs.WindowStyle = "Hidden"
    }
    $proc = Start-Process @startArgs
    $code = $proc.ExitCode
    Add-Content -LiteralPath $LogFile -Value "----- $(Get-Date -Format o) exit=$code -----" -Encoding utf8
    if (Test-Path -LiteralPath $stdoutFile) {
        Get-Content -LiteralPath $stdoutFile | Add-Content -LiteralPath $LogFile -Encoding utf8
    }
    if (Test-Path -LiteralPath $stderrFile) {
        Get-Content -LiteralPath $stderrFile | Add-Content -LiteralPath $LogFile -Encoding utf8
    }

    if ($code -eq 0) {
        Write-Log "training exited 0; target $TargetUpdates updates reached. Stopping."
        exit 0
    }
    if ($code -eq 2) {
        Write-Log "training exited 2 (config hash mismatch). Not retrying. Fix the config or pass -AllowConfigMismatch."
        exit 2
    }
    Write-Log "training exited $code. Waiting $RetrySeconds s, then --resume auto."
    Start-Sleep -Seconds $RetrySeconds
}
