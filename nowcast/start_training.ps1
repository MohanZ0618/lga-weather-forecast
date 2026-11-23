$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw "Missing project Python at $python" }
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$runDir = Join-Path $PSScriptRoot "training_runs/$runId"
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null
$env:LOKY_MAX_CPU_COUNT = '2'
$process = Start-Process -FilePath $python -WorkingDirectory $project -WindowStyle Hidden -PassThru `
    -ArgumentList @('-u', 'nowcast/train_local.py', '--run-dir', "nowcast/training_runs/$runId") `
    -RedirectStandardOutput (Join-Path $runDir 'training.log') `
    -RedirectStandardError (Join-Path $runDir 'errors.log')
[pscustomobject]@{ ProcessId = $process.Id; RunDirectory = $runDir }
