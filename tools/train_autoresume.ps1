#Requires -Version 5.1
<#
.SYNOPSIS
    Restart transformer PPO until train.num_updates is finished.

.DESCRIPTION
    Each attempt asks tools/train_launch_mode.py which command to run.

    fresh: the checkpoint directory has no complete PPO update_*.pt.
    The training command is --init-weights <BcCheckpoint>
    --allow-cross-backend-init. That loads BC weights only, not the BC Adam
    state.

    resume: at least one complete PPO checkpoint exists (same readability
    check as --resume auto). The command is only --resume auto. A corrupt
    newest file with an older complete file is still resume. --init-weights
    is not passed, so a restart cannot reload the BC weights.

    The decision is repeated after every non-zero exit. A non-zero training
    exit waits RetrySeconds (default 30) and tries again. Exit 0 means the
    target update count was reached and this script stops. Exit 2 is a
    config-hash refusal, or a fresh start whose BC file is missing, and is
    not retried.

    The target is -TargetUpdates when that is greater than 0, otherwise
    train.num_updates in -Config. Paths are resolved from the repo root
    (the parent of this tools directory), not from the scheduler's
    working directory.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,

    [Parameter(Mandatory = $true)]
    [string]$BcCheckpoint,

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
$BcPath = Resolve-RepoPath $BcCheckpoint

$payload = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($TargetUpdates -le 0) {
    if (-not $payload.train -or -not $payload.train.num_updates) {
        throw "Config $ConfigPath has no train.num_updates. Pass -TargetUpdates."
    }
    $TargetUpdates = [int]$payload.train.num_updates
}

function Get-ExtraValue([string]$Name) {
    $found = $null
    if (-not $ExtraArgs) {
        return $null
    }
    for ($i = 0; $i -lt $ExtraArgs.Count; $i++) {
        if ($ExtraArgs[$i] -eq $Name -and ($i + 1) -lt $ExtraArgs.Count) {
            $found = $ExtraArgs[$i + 1]
        }
    }
    return $found
}

$checkpointOverride = Get-ExtraValue "--checkpoint-dir"
if ($checkpointOverride) {
    $CheckpointDir = Resolve-RepoPath ([string]$checkpointOverride)
} else {
    if (-not $payload.train -or -not $payload.train.checkpoint_dir) {
        throw "Config $ConfigPath has no train.checkpoint_dir. Pass -ExtraArgs --checkpoint-dir <dir>."
    }
    $CheckpointDir = Resolve-RepoPath ([string]$payload.train.checkpoint_dir)
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
$launchHelper = Join-Path $RepoRoot "tools\train_launch_mode.py"

Write-Log "autoresume start target_updates=$TargetUpdates config=$ConfigPath bc=$BcPath checkpoint_dir=$CheckpointDir"

while ($true) {
    $modeText = & $Python $launchHelper --checkpoint-dir $CheckpointDir
    $helperCode = $LASTEXITCODE
    if ($helperCode -ne 0) {
        Write-Log "launch-mode helper exited $helperCode. Waiting $RetrySeconds s."
        Start-Sleep -Seconds $RetrySeconds
        continue
    }
    $modeLines = @(
        @($modeText) | Where-Object { $_ -ne $null -and "$_".Trim() -ne "" }
    )
    if ($modeLines.Count -lt 1) {
        $mode = ""
    } else {
        $mode = "$($modeLines[-1])".Trim()
    }
    $argList = @(
        $trainScript,
        "--config", $ConfigPath,
        "--updates", "$TargetUpdates"
    )
    if ($mode -eq "resume") {
        $argList += @("--resume", "auto")
        Write-Log "launch mode resume: --resume auto (init-weights not passed)"
    } elseif ($mode -eq "fresh") {
        if (-not (Test-Path -LiteralPath $BcPath)) {
            Write-Log "fresh start needs a BC checkpoint and $BcPath is missing. Not retrying."
            exit 2
        }
        $argList += @("--init-weights", $BcPath, "--allow-cross-backend-init")
        Write-Log "launch mode fresh: --init-weights (no --resume)"
    } else {
        Write-Log "launch-mode helper printed '$modeText'. Waiting $RetrySeconds s."
        Start-Sleep -Seconds $RetrySeconds
        continue
    }
    if ($AllowConfigMismatch) {
        $argList += "--allow-config-mismatch"
    }
    if ($ExtraArgs) {
        $argList += $ExtraArgs
    }

    $argumentString = ($argList | ForEach-Object { Quote-ProcessArg $_ }) -join " "
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdoutFile = Join-Path $logDir "train-$stamp.out.txt"
    $stderrFile = Join-Path $logDir "train-$stamp.err.txt"
    Write-Log "command: $Python $argumentString"

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
    Write-Log "training exited $code. Waiting $RetrySeconds s, then choosing fresh or resume again."
    Start-Sleep -Seconds $RetrySeconds
}
