@echo off
rem run.bat : runs a Python script with the project's Python environment.
rem Usage (PowerShell, from this folder):   .\run evaluate.py cs1
rem It uses the first Python it finds:
rem   1. .venv\ in this folder (see "Setup" in README.md)
rem   2. ..\rlpid\rlpid-env\ (the original working environment)
rem   3. "python" on the PATH
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" %*
) else if exist "%~dp0..\rlpid\rlpid-env\Scripts\python.exe" (
    "%~dp0..\rlpid\rlpid-env\Scripts\python.exe" %*
) else (
    python %*
)
