param([string]$Python = "", [int]$BatchSize = 16, [ValidateSet("auto", "cuda", "cpu")][string]$Device = "auto")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\python.ps1"
$Interpreter = Get-CastlePython $Python
& $Interpreter -m castle build --batch-size $BatchSize --device $Device
if ($LASTEXITCODE -ne 0) { throw "Tao index that bai. Ket qua embedding da hoan thanh van duoc giu de chay tiep." }
