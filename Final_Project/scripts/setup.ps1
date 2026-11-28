param([string]$Python = "")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Python) {
    $Candidates = @("D:\UIT\.venvs\ml-gpu\Scripts\python.exe", "D:\UIT\ml-gpu\Scripts\python.exe")
    foreach ($Candidate in $Candidates) { if (Test-Path $Candidate) { $Python = $Candidate; break } }
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw 'Khong tim thay ml-gpu. Chay lai: .\scripts\setup.ps1 -Python "D:\duong-dan\ml-gpu\Scripts\python.exe"'
}
New-Item -ItemType Directory -Force .runtime | Out-Null
$env:PIP_CACHE_DIR = Join-Path (Get-Location).Path ".cache\pip"
& $Python -m pip freeze | Set-Content -Encoding UTF8 .runtime\environment_before.txt
if ($LASTEXITCODE -ne 0) { throw "Khong doc duoc pip trong venv." }
# Protect the GPU stack already installed by the user. Fail rather than silently replace it.
& $Python .\scripts\preserve_environment.py
if ($LASTEXITCODE -ne 0) { throw "Chua xac nhan duoc torch; khong cai tiep." }
$Python | Set-Content -Encoding UTF8 .runtime\python_path.txt
# Pip must respect existing torch/vision/audio/numpy versions. No --upgrade.
& $Python -m pip install -r requirements.txt -c .runtime\gpu_constraints.txt
if ($LASTEXITCODE -ne 0) { throw "Cai dependency that bai. Giu nguyen venv, gui lai loi; khong tu ha torch." }
& $Python -m castle doctor
if ($LASTEXITCODE -ne 0) { throw "Doctor phat hien loi. Xu ly loi moi truong/du lieu truoc khi build." }
Write-Host "`nSetup xong. Tiep theo: .\scripts\build_index.ps1" -ForegroundColor Green
