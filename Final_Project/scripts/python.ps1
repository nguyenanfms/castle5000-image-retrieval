function Get-CastlePython {
    param([string]$Requested = "")
    if ($Requested) { if (-not (Test-Path $Requested)) { throw "Python path khong ton tai." }; return $Requested }
    $SavedPath = Join-Path (Split-Path $PSScriptRoot -Parent) ".runtime\python_path.txt"
    if (Test-Path $SavedPath) {
        $Candidate = (Get-Content -Raw $SavedPath).Trim()
        if (Test-Path $Candidate) { return $Candidate }
    }
    foreach ($Candidate in @("D:\UIT\.venvs\ml-gpu\Scripts\python.exe", "D:\UIT\ml-gpu\Scripts\python.exe")) {
        if (Test-Path $Candidate) { return $Candidate }
    }
    throw "Khong tim thay Python. Chay setup.ps1 truoc."
}
