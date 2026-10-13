@echo off
rem Builds dist\MeetingScribe\ (MeetingScribe.exe + the Chrome extension folder): zip it to share
cd /d %~dp0
.venv\Scripts\python.exe -m PyInstaller MeetingScribe.spec --noconfirm --clean
xcopy /E /I /Y extension dist\MeetingScribe\extension >nul
