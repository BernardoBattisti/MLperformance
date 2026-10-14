@echo off
REM Builds dist\MLperformance\MLperformance.exe and a zip ready to share in release\.
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv || goto :error
    .venv\Scripts\python -m pip install -r requirements.txt || goto :error
)
.venv\Scripts\python -m pip install "pyinstaller>=6.10" || goto :error
.venv\Scripts\python -m PyInstaller --noconfirm --clean MLperformance.spec || goto :error
echo Verifying the build...
dist\MLperformance\MLperformance.exe --self-test || goto :error
if not exist release mkdir release
powershell -NoProfile -Command "Compress-Archive -Path dist\MLperformance -DestinationPath release\MLperformance-windows.zip -Force" || goto :error
echo.
echo Done: dist\MLperformance\MLperformance.exe
echo       release\MLperformance-windows.zip
goto :eof

:error
echo Build failed.
exit /b 1
