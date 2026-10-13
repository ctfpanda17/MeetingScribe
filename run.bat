@echo off
cd /d %~dp0
set PYTHONIOENCODING=utf-8
start "" .venv\Scripts\pythonw.exe app\main.py
