@echo off
setlocal enabledelayedexpansion

REM ==================================================
REM == Step -1: Check Administrator Privileges ==
REM ==================================================
echo Checking for Administrator privileges...
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo ERROR: This script requires Administrator privileges.
    echo Please right-click the script and select 'Run as administrator'.
    goto :error_no_endlocal
) else (
    echo Administrator privileges detected. Proceeding...
)
echo.

REM ==================================================
REM == Main Installation Logic Starts Here ==
REM ==================================================

REM Define the ABSOLUTE project directory path in user's home folder
set USER_HOME=%USERPROFILE%
set TARGET_INSTALL_DIR=%USER_HOME%\.sagax1\web-ui
set TEMP_DIR=%TEMP%\sagax1_installer
set PYTHON_EMBED_DIR=%TARGET_INSTALL_DIR%\python_embed
set DOWNLOAD_DIR=%TEMP_DIR%\downloads

echo Target installation directory set to: %TARGET_INSTALL_DIR%
echo.

REM Check if the target directory already exists
if exist "%TARGET_INSTALL_DIR%\" (
    echo Directory "%TARGET_INSTALL_DIR%" already exists.
    echo Previous installation detected. Proceeding will update existing files.
    choice /C YN /M "Do you want to continue?"
    if errorlevel 2 goto :error
)

REM Create temporary directory for downloads
echo Creating temporary directory for downloads...
mkdir "%TEMP_DIR%" 2>nul
mkdir "%DOWNLOAD_DIR%" 2>nul

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

REM == Step 1: Download and Extract Repository ZIP ==
echo Downloading repository ZIP file...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/khurrameycon/update-webui/archive/refs/heads/main.zip' -OutFile '%DOWNLOAD_DIR%\repo.zip'}"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download repository ZIP.
    echo Check your internet connection.
    goto :error
)

echo Extracting repository...
powershell -Command "& {Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory('%DOWNLOAD_DIR%\repo.zip', '%TEMP_DIR%')}"
if %errorlevel% neq 0 (
    echo ERROR: Failed to extract repository ZIP.
    goto :error
)

echo Copying repository files to target directory...
xcopy /E /I /Y "%TEMP_DIR%\update-webui-main\*" "%TARGET_INSTALL_DIR%\"
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy repository files.
    goto :error
)

echo Repository files copied successfully.
echo.

REM Change directory AND drive to the target installation directory
cd /d "%TARGET_INSTALL_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to change directory to "%TARGET_INSTALL_DIR%".
    goto :error
)
echo Current directory is now: "%CD%"
echo.

REM == Step 2: Download and Setup Python Embedded ==
echo Downloading Python embedded...
mkdir "%PYTHON_EMBED_DIR%" 2>nul
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.3/python-3.13.3-embed-amd64.zip' -OutFile '%DOWNLOAD_DIR%\python_embed.zip'}"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download Python embedded.
    echo Check your internet connection.
    goto :error
)

echo Extracting Python embedded...
REM Clear the directory before extraction to avoid conflicts
if exist "%PYTHON_EMBED_DIR%" (
    echo Cleaning existing Python embedded directory...
    rd /s /q "%PYTHON_EMBED_DIR%"
    mkdir "%PYTHON_EMBED_DIR%"
)
powershell -Command "& {Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory('%DOWNLOAD_DIR%\python_embed.zip', '%PYTHON_EMBED_DIR%')}"
if %errorlevel% neq 0 (
    echo ERROR: Failed to extract Python embedded.
    goto :error
)

echo Python embedded extracted successfully.
echo.

REM Ensure pip is available in embedded Python
echo Setting up pip for embedded Python...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%DOWNLOAD_DIR%\get-pip.py'}"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download get-pip.py.
    goto :error
)

REM Fix python**._pth file to allow importing site packages
echo Modifying Python path configuration...
set PTH_FILE=
for %%f in ("%PYTHON_EMBED_DIR%\python*._pth") do set PTH_FILE=%%f
if "%PTH_FILE%"=="" (
    echo ERROR: Could not find python._pth file.
    goto :error
)

REM Backup original file
copy /Y "%PTH_FILE%" "%PTH_FILE%.bak" >nul

REM Modify the file to uncomment import site
type "%PTH_FILE%" | findstr /v /c:"import site" > "%TEMP%\pth.tmp"
echo import site >> "%TEMP%\pth.tmp"
copy /Y "%TEMP%\pth.tmp" "%PTH_FILE%" >nul
del "%TEMP%\pth.tmp"

REM Download and install pip using get-pip.py
echo Downloading get-pip.py...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%DOWNLOAD_DIR%\get-pip.py'}"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download get-pip.py.
    goto :error
)

echo Configuring Python libraries path...
mkdir "%PYTHON_EMBED_DIR%\Lib" 2>nul
mkdir "%PYTHON_EMBED_DIR%\Lib\site-packages" 2>nul

echo Running get-pip.py with embedded Python...
"%PYTHON_EMBED_DIR%\python.exe" "%DOWNLOAD_DIR%\get-pip.py" --no-warn-script-location
if %errorlevel% neq 0 (
    echo ERROR: Failed to install pip.
    goto :error
)

echo Installing necessary Python packages...
echo Adding Scripts directory to PATH temporarily...
set PATH=%PYTHON_EMBED_DIR%;%PYTHON_EMBED_DIR%\Scripts;%PATH%

echo Installing virtualenv with embedded Python...
"%PYTHON_EMBED_DIR%\python.exe" -m pip install virtualenv --no-warn-script-location
if %errorlevel% neq 0 (
    echo ERROR: Failed to install virtualenv.
    goto :error
)
echo.

REM == Step 3: Create Python Virtual Environment ==
echo Creating Python virtual environment (.venv) in "%CD%"...
"%PYTHON_EMBED_DIR%\Scripts\virtualenv.exe" .venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
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

REM Create a modified requirements file without the problematic dependency
echo Creating a modified requirements file...
type NUL > requirements_modified.txt
for /F "tokens=*" %%a in (requirements.txt) do (
    echo %%a | findstr /i "langchain-ibm" > nul
    if errorlevel 1 (
        echo %%a >> requirements_modified.txt
    ) else (
        echo Skipping incompatible package: %%a
        echo # %%a >> requirements_modified.txt
    )
)

echo Installing compatible packages...
python -m pip install --no-cache-dir -r requirements_modified.txt
set PIP_RESULT=%errorlevel%

REM Process the error but don't exit
if %PIP_RESULT% neq 0 (
    echo WARNING: Some packages failed to install.
    echo Attempting to install essential packages individually...
    
    REM Try to install essential packages one by one
    python -m pip install --no-cache-dir gradio
    python -m pip install --no-cache-dir langchain
    python -m pip install --no-cache-dir playwright
    
    echo Installation of core packages completed with best effort.
    echo Some features may not be available due to missing dependencies.
) else (
    echo Python packages installed successfully.
)
echo.

REM == Step 5: Install Playwright Browsers ==
echo Installing Playwright browsers (this might take a while)...
echo Attempting to install Playwright with Chromium...
python -m pip install playwright
if %errorlevel% neq 0 (
    echo WARNING: Failed to install Playwright package.
    echo Browser automation features may not work properly.
) else (
    playwright install --with-deps chromium
    if %errorlevel% neq 0 (
        echo WARNING: Playwright browser installation reported an error.
        echo Browser automation features may not work properly.
    ) else (
        echo Playwright browsers installed/updated successfully.
    )
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
echo    - Navigate to this directory: cd /d "%TARGET_INSTALL_DIR%"
echo    - Double-click on the 'start_app.bat' file we'll create for you
echo.
echo 3. Access the Web UI in your browser, usually at: http://127.0.0.1:7788
echo.

REM Create a simple startup script for the user
echo Creating startup script...
(
    echo @echo off
    echo cd /d "%TARGET_INSTALL_DIR%"
    echo call .venv\Scripts\activate.bat
    echo echo Starting Web UI...
    echo python webui.py
    echo pause
) > "%TARGET_INSTALL_DIR%\start_app.bat"
echo Please Launch the Visual Web Agent by Clicking Launch Button
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
rmdir /s /q "%TEMP_DIR%" 2>nul
echo Script finished successfully.
pause
endlocal
exit /b 0

:cleanup_and_exit
echo Cleaning up temporary downloads...
rmdir /s /q "%TEMP_DIR%" 2>nul
endlocal
exit /b 1