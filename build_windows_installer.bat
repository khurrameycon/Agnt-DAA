@echo off
setlocal enabledelayedexpansion

echo ================================================
echo sagax1 Windows Installer Build Script
echo ================================================

:: Create build directory
if not exist build mkdir build
if not exist dist mkdir dist

:: Step 1: Check Python installation
echo.
echo Step 1: Checking Python installation...
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.8 or later.
    goto :error
) else (
    for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
    echo Python !PYTHON_VERSION! found.
)

:: Step 2: Set up virtual environment
echo.
echo Step 2: Setting up virtual environment...
if not exist venv (
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 goto :error
    echo Created new virtual environment.
) else (
    echo Using existing virtual environment.
)

:: Activate virtual environment
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment.
    goto :error
)
echo Virtual environment activated.

:: Step 3: Install dependencies
echo.
echo Step 3: Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 goto :error
pip install pyinstaller pillow
if %ERRORLEVEL% NEQ 0 goto :error
echo Dependencies installed successfully.

:: Step 4: Check for NSIS
echo.
echo Step 4: Checking for NSIS...
python check_nsis.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: NSIS check failed. Installer creation might fail.
) else (
    echo NSIS check completed.
)

:: Step 5: Create icon file
echo.
echo Step 5: Creating application icon...
python create_icon.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Failed to create icon. Using default icon.
)

:: Create simple license file if it doesn't exist
if not exist LICENSE.txt (
    echo.
    echo Creating sample LICENSE.txt file...
    echo sagax1 Software License Agreement > LICENSE.txt
    echo Copyright (c) 2025 sagax1 Team >> LICENSE.txt
    echo All rights reserved. >> LICENSE.txt
)

:: Step 6: Create installer graphics
echo.
echo Step 6: Creating installer graphics...
python create_installer_graphics.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Failed to create installer graphics. Using defaults.
)

:: Step 7: Build executable
echo.
echo Step 7: Building executable with PyInstaller...
pyinstaller --noconfirm main.spec
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyInstaller build failed.
    goto :error
)

:: Step 8: Check if build was successful
if not exist dist\sagax1 (
    echo ERROR: Build failed, dist\sagax1 directory not found.
    goto :error
)

:: Step 9: Build NSIS installer
echo.
echo Step 9: Building NSIS installer...
makensis installer.nsi
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: NSIS installer creation failed.
    echo You can still run the application from dist\sagax1\sagax1.exe
)

:: Look for the installer
if exist dist\sagax1_Setup.exe (
    echo.
    echo ================================================
    echo Build completed successfully!
    echo.
    echo Installer created at: dist\sagax1_Setup.exe
    echo ================================================
) else (
    echo.
    echo ================================================
    echo Build partially completed.
    echo.
    echo Executable created in dist\sagax1 folder
    echo but installer was not created.
    echo.
    echo You can run the application from dist\sagax1\sagax1.exe
    echo ================================================
)

goto :end

:error
echo.
echo ================================================
echo Build failed!
echo ================================================

:end
:: Deactivate virtual environment
call venv\Scripts\deactivate.bat
echo.
echo Press any key to exit...
pause > nul