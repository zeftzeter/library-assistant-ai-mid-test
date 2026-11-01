param([switch]$Demo, [string]$UserId = '', [string]$Ask = '')
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:LANGSMITH_TRACING = 'false'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv and install requirements first. See README.md.'
}
$agentArgs = @((Join-Path $PSScriptRoot 'library_agent.py'))
if ($UserId) { $agentArgs += @('--user', $UserId) }
if ($Demo) { $agentArgs += '--demo' }
if ($Ask) { $agentArgs += @('--ask', $Ask) }
& $pythonPath @agentArgs
exit $LASTEXITCODE
