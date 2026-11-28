param([string]$Python = "", [int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\python.ps1"
$Interpreter = Get-CastlePython $Python
Write-Host "Mo trinh duyet: http://127.0.0.1:$Port"
Write-Host "Ctrl+C de dung server."
& $Interpreter -m castle serve --port $Port
