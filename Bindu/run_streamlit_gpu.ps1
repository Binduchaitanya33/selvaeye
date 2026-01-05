# Runs SafetyEye Streamlit app using the 'gpu' conda environment
# (ensures ultralytics is available)

$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

$condaExe = $env:CONDA_EXE
if (-not $condaExe -or -not (Test-Path $condaExe)) {
  $condaExe = Join-Path $HOME 'anaconda3\Scripts\conda.exe'
}

if (-not (Test-Path $condaExe)) {
  throw "conda.exe not found. Install Anaconda/Miniconda or set CONDA_EXE."
}

& $condaExe run -n gpu --no-capture-output python -m streamlit run app.py
