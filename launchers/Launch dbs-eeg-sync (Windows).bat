@echo off
REM Double-click launcher for Windows.
REM
REM What it does, in plain terms:
REM   1. Finds the project folder (the folder this file lives in, one level up).
REM   2. Makes sure a Python environment with the toolbox installed exists
REM      (creates one the first time, which can take a few minutes).
REM   3. Opens the graphical EEG/DBS synchronization window.
REM
REM A clinician should be able to just double-click this file in Explorer.

setlocal

REM Project root = the folder containing this "launchers" folder.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."
set "PROJECT_DIR=%CD%"

set "VENV_DIR=%PROJECT_DIR%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

REM Find a system Python to bootstrap with.
where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on this PC.
  echo Please install Python 3.12+ from https://www.python.org/downloads/
  echo Make sure to tick "Add Python to PATH" during installation.
  pause
  popd
  exit /b 1
)

REM Create the environment and install the toolbox on first run.
if not exist "%PYTHON%" (
  echo First-time setup: creating a Python environment ^(this may take a few minutes^)...
  python -m venv "%VENV_DIR%"
  "%PYTHON%" -m pip install --upgrade pip
  "%PYTHON%" -m pip install -e ".[gui]"
)

REM Make sure the GUI components are present.
"%PYTHON%" -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
  echo Installing the graphical-interface components...
  "%PYTHON%" -m pip install -e ".[gui]"
)

REM Launch the graphical window.
"%PYTHON%" -m dbs_eeg_sync

popd
endlocal
