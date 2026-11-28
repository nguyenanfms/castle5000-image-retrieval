param([string]$Python = "")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\python.ps1"
$Interpreter = Get-CastlePython $Python
& $Interpreter -m castle doctor
if ($LASTEXITCODE -ne 0) { throw "Doctor phat hien loi. Xem chan doan o tren." }
