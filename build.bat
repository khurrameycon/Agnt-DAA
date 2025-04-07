@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo sagax1 Windows Installer Build System
echo ================================================================
echo This script will guide you through the process of building
echo the sagax1 application and creating a Windows installer.
echo.

:: Check Python
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8 or later and try again.
    goto :end
)

:: Options
echo Build options:
echo 1. Full build with installer (recommended)
echo 2. Package application without installer
echo 3. Check environment and project structure only
echo 4. Exit
echo.

set /p choice=Enter your choice (1-4): 

if "%choice%"=="1" (
    call :check_environment
    if !ERRORLEVEL! NEQ 0 goto :end
    
    call :check_structure
    if !ERRORLEVEL! NEQ 0 goto :end
    
    call :full_build
    if !ERRORLEVEL! NEQ 0 goto :end
) else if "%choice%"=="2" (
    call :check_environment
    if !ERRORLEVEL! NEQ 0 goto :end
    
    call :check_structure
    if !ERRORLEVEL! NEQ 0 goto :end
    
    call :package_only
    if !ERRORLEVEL! NEQ 0 goto :end
) else if "%choice%"=="3" (
    call :check_environment
    if !ERRORLEVEL! NEQ 0 goto :end
    
    call :check_structure
    if !ERRORLEVEL! NEQ 0 goto :end
    
    echo.
    echo Environment and project structure checks passed.
) else if "%choice%"=="4" (
    goto :end
) else (
    echo Invalid choice. Please select 1-4.
    goto :end
)

goto :end

:: Check environment
:check_environment
echo.
echo Checking build environment...

:: Check Python version
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo Python !PYTHON_VERSION! found.

:: Check pip
python -m pip --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip is not installed or not working.
    exit /b 1
)

:: Check venv
python -c "import venv" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: venv module is not available.
    echo Please run: pip install venv
    exit /b 1
)

:: Success
echo Build environment check passed.
exit /b 0

:: Check project structure
:check_structure
echo.
echo Checking project structure...
python check_project_structure.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Project structure check failed.
    exit /b 1
)
exit /b 0

:: Full build with installer
:full_build
echo.
echo Starting full build with installer...
call build_windows_installer.bat
exit /b %ERRORLEVEL%

:: Package only
:package_only
echo.
echo Packaging application without installer...
call package_app.bat
exit /b %ERRORLEVEL%

:end
echo.
echo Build process completed.
pause