# UI translations (Qt style): scan ui.py for tr("...") strings and keep locales\<lang>.json in step.
# Usage: i18n.ps1 update [--prune] | check | new <lang>        (main env; i18n.py)
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $MainPython)) { throw 'Main environment is missing: run.bat setup-all --main-only (or setup-all)' }
Invoke-Checked $MainPython (@('-u', (Join-Path $ProjectRoot 'i18n.py')) + $Extra)
