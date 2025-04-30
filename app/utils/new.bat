@echo off
setlocal enabledelayedexpansion

REM ==================================================
REM == Step -1: Check Administrator Privileges ==
REM ==================================================
echo Checking for Administrator privileges...
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo ERROR: This script requires Administrator privileges to install Git and Python.
    echo Please right-click the script and select 'Run as administrator'.
    goto :error_no_endlocal
) else (
    echo Administrator privileges detected. Proceeding...
)
echo.

@echo off
setlocal enabledelayedexpansion

echo Checking for Git and Python installations...
echo.

:: Check for Git
where git >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=3" %%a in ('git --version 2^>^&1') do set "GIT_VERSION=%%a"
    echo Git is already installed. Version: !GIT_VERSION!
) else (
    echo Git is not installed. Downloading Git...
    
    :: Create temp directory for downloads
    mkdir "%TEMP%\installer_downloads" 2>nul
    
    :: Download Git
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.1/Git-2.41.0-64-bit.exe' -OutFile '%TEMP%\installer_downloads\git_installer.exe'}"
    
    if exist "%TEMP%\installer_downloads\git_installer.exe" (
        echo Installing Git...
        start /wait "" "%TEMP%\installer_downloads\git_installer.exe" /VERYSILENT /NORESTART
        echo Git has been installed.
    ) else (
        echo Failed to download Git installer.
    )
)

:: Check for Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=2" %%a in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%a"
    echo Python is already installed. Version: !PYTHON_VERSION!
) else (
    echo Python is not installed. Downloading Python...
    
    :: Create temp directory for downloads if it doesn't exist
    mkdir "%TEMP%\installer_downloads" 2>nul
    
    :: Download Python
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe' -OutFile '%TEMP%\installer_downloads\python_installer.exe'}"
    
    if exist "%TEMP%\installer_downloads\python_installer.exe" (
        echo Installing Python...
        start /wait "" "%TEMP%\installer_downloads\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1
        echo Python has been installed.
    ) else (
        echo Failed to download Python installer.
    )
)

:: Check if PATH needs to be refreshed
echo.
echo Installation completed. You may need to restart your command prompt to use Git or Python commands.
echo.

:: Verify installations
echo Verifying installations:
echo.

:: Verify Git
where git >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Git: INSTALLED
) else (
    echo Git: NOT FOUND - You may need to restart your command prompt or rerun this script.
)

:: Verify Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Python: INSTALLED
) else (
    echo Python: NOT FOUND - You may need to restart your command prompt or rerun this script.
)



REM ==================================================
REM == Main Installation Logic Starts Here ==
REM ==================================================

REM Define the ABSOLUTE project directory path in user's home folder
set USER_HOME=%USERPROFILE%
set TARGET_INSTALL_DIR=%USER_HOME%\.sagax1\web-ui

echo Target installation directory set to: %TARGET_INSTALL_DIR%
echo.

REM Check if the target directory already exists
if exist "%TARGET_INSTALL_DIR%\" (
    echo Directory "%TARGET_INSTALL_DIR%" already exists.
    echo Previous installation detected. Proceeding will update existing files.
    choice /C YN /M "Do you want to continue?"
    if errorlevel 2 goto :error
)

REM Ensure the parent directory exists
if not exist "%USER_HOME%\.sagax1\" (
    echo Creating .sagax1 directory in user home folder...
    mkdir "%USER_HOME%\.sagax1"
)

REM Attempt to create the target directory if it doesn't exist
if not exist "%TARGET_INSTALL_DIR%\" (
    echo Creating target directory: %TARGET_INSTALL_DIR%
    mkdir "%TARGET_INSTALL_DIR%"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create directory "%TARGET_INSTALL_DIR%".
        echo Check permissions or if the path is valid.
        goto :error
    )
    echo Target directory created.
) else (
    echo Target directory already exists.
)
echo.


REM == Step 1: Verify Prerequisites Again (Post Step 0) ==
echo Verifying prerequisites again before proceeding...
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Git not found even after Step 0. Please check installation or restart Command Prompt and retry.
    goto :error
) else (
    echo Found Git.
)

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found even after Step 0. Please check installation or restart Command Prompt and retry.
    goto :error
) else (
    echo Found Python.
    python --version
)
echo.

REM == Step 2: Clone the Repository ==
echo Cloning the web-ui repository into %TARGET_INSTALL_DIR%...
git clone https://github.com/khurrameycon/web-ui.git "%TARGET_INSTALL_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to clone repository. Check your internet connection and Git setup.
    goto :error
)

REM Change directory AND drive to the target installation directory
cd /d "%TARGET_INSTALL_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to change directory to "%TARGET_INSTALL_DIR%".
    goto :error
)
echo Successfully cloned repository and changed directory to "%CD%".
echo.

REM == Step 3: Create Python Virtual Environment ==
echo Creating Python virtual environment (.venv) in "%CD%"...
python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment. Check your Python installation.
    goto :error
)
echo Virtual environment created.
echo.

REM == Step 4: Activate Environment and Install Dependencies ==
echo Activating virtual environment and installing Python packages...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment. Check path: "%CD%\.venv\Scripts\activate.bat"
    goto :error
)

echo Installing packages from requirements.txt (using --no-cache-dir)...
python -m pip install --no-cache-dir -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python packages. Check requirements.txt and pip setup.
    goto :error
)
echo Python packages installed.
echo.

REM == Step 5: Install Playwright Browsers ==
echo Installing Playwright browsers (this might take a while)...
playwright install --with-deps chromium
REM You can change 'chromium' to install others or remove '--with-deps chromium' to install all default browsers
if %errorlevel% neq 0 (
    echo WARNING: Playwright browser installation reported an error. The UI might still work,
    echo but browser automation could fail. Try running 'playwright install' manually later.
    REM Don't exit on playwright error, maybe user wants to retry manually
) else (
    echo Playwright browsers installed/updated.
)
echo.

REM == Step 6: Setup Configuration File ==
echo Setting up configuration file...
if not exist ".env.example" (
    echo ERROR: .env.example not found in the repository. Cannot create .env file.
    goto :error
)
copy .env.example .env
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy .env.example to .env.
    goto :error
)
echo Copied .env.example to .env.
echo.

REM == Step 7: Installation Complete - Instructions ==
echo ==================================================
echo  Installation process completed!
echo ==================================================
echo.
echo NEXT STEPS:
echo 1. IMPORTANT: Manually edit the '.env' file in the '%TARGET_INSTALL_DIR%' directory.
echo    You MUST add your API keys (OpenAI, Google, Anthropic, etc.) for the AI features to work.
echo.
echo 2. To run the WebUI:
echo    - Open a new Command Prompt or PowerShell window (important if Git/Python were just installed).
echo    - Navigate to this directory: cd /d "%TARGET_INSTALL_DIR%"
echo    - Activate the virtual environment: .venv\Scripts\activate.bat
echo    - Run the application: python webui.py
echo.
echo 3. Access the WebUI in your browser, usually at: http://127.0.0.1:7788
echo.
goto :cleanup_and_end

:error
echo --------------------------------------------------
echo  Installation failed. Please check the error messages above.
echo --------------------------------------------------
pause
goto :cleanup_and_exit

:error_no_endlocal
REM Separate error exit for early failure before endlocal is appropriate
echo Script halted.
pause
exit /b 1

:cleanup_and_end
echo Cleaning up temporary downloads...
rmdir /s /q "%DOWNLOAD_DIR%" 2>nul
echo Script finished successfully.
pause
endlocal
exit /b 0

:cleanup_and_exit
echo Cleaning up temporary downloads...
rmdir /s /q "%DOWNLOAD_DIR%" 2>nul
endlocal
exit /b 1