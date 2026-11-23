# One command for Windows: create .venv on first use, install requirements, then start the app.
#   powershell -ExecutionPolicy Bypass -File .\start.ps1            (demo DRES, port 8000)
#   powershell -ExecutionPolicy Bypass -File .\start.ps1 --real-dres (DRES from .env)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host 'Creating .venv (Python 3.13)...'
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.13 -m venv .venv }
    else { python -m venv .venv }
    if (-not (Test-Path -LiteralPath $python)) { throw 'Could not create .venv. Install Python 3.13 (64-bit) first.' }
}
$marker = Join-Path $PSScriptRoot '.venv\.requirements-installed'
$stamp = (Get-FileHash -LiteralPath 'requirements.txt').Hash
if (-not (Test-Path -LiteralPath $marker) -or (Get-Content -LiteralPath $marker) -ne $stamp) {
    Write-Host 'Installing requirements (first run takes a while)...'
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
    Set-Content -LiteralPath $marker -Value $stamp -Encoding ascii
}
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host 'Created .env from .env.example. Add your Gemini keys there to enable QA and Gemini checks.'
}
$env:PYTHONUTF8 = '1'
& $python tools/start_local.py @args
exit $LASTEXITCODE
