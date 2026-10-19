@echo off
:: The only batch file: a launcher so that double-click / drag-and-drop works without touching
:: the PowerShell execution policy. All logic lives in run.ps1 and scripts\*.ps1.
where pwsh >nul 2>&1 && (set "PS=pwsh") || (set "PS=powershell")
%PS% -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
pause
