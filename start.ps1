$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    py -3.11 -m venv .venv
}
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости' }
& '.venv/Scripts/python.exe' scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Не удалось загрузить модель' }
& '.venv/Scripts/python.exe' scripts/serve.py
