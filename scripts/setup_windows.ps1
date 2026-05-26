param(
    [string]$PythonExe = ".\tools\Python310\python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (!(Test-Path $PythonExe)) {
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $systemPython) {
        throw "Python not found. Install Python 3.10+ or place it at tools\Python310\python.exe."
    }
    $PythonExe = $systemPython.Source
}

& $PythonExe -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Environment ready."
