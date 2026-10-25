@echo off
setlocal
cd /d "%~dp0"

rem Find uv on PATH or in its default install location.
where uv >nul 2>&1 && goto :run
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    goto :run
)

echo uv not found, installing...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 (
    echo Failed to install uv. See https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>&1 || (
    echo uv was installed but could not be found. Open a new terminal and try again.
    pause
    exit /b 1
)

:run
uv run starfield %*
if errorlevel 1 pause
exit /b %errorlevel%
