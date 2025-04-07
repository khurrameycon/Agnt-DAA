@echo off
echo ================================================
echo sagax1 Simple Packaging Script
echo ================================================
echo This script will package the application without creating an installer.
echo Useful for testing or for portable distribution.

:: Create build directory
if not exist build mkdir build
if not exist dist mkdir dist

:: Step 1: Set up virtual environment
echo.
echo Step 1: Setting up virtual environment...
if not exist venv (
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 goto :error
) else (
    echo Using existing virtual environment.
)

:: Activate virtual environment
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 goto :error

:: Step 2: Install dependencies
echo.
echo Step 2: Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 goto :error
pip install pyinstaller pillow
if %ERRORLEVEL% NEQ 0 goto :error

:: Step 3: Create icon file
echo.
echo Step 3: Creating application icon...
python create_icon.py

:: Step 4: Build executable using PyInstaller
echo.
echo Step 4: Building executable with the "onefile" option...
pyinstaller --noconfirm --clean ^
    --name=sagax1_Portable ^
    --icon=assets\icons\sagax1-logo.ico ^
    --add-data="assets;assets" ^
    --add-data="config;config" ^
    --hidden-import=smolagents ^
    --hidden-import=huggingface_hub ^
    --hidden-import=transformers ^
    --hidden-import=torch ^
    --hidden-import=app ^
    --hidden-import=app.ui ^
    --hidden-import=app.core ^
    --hidden-import=app.utils ^
    --hidden-import=app.agents ^
    --hidden-import=selenium ^
    --hidden-import=helium ^
    --hidden-import=webdriver_manager ^
    --noconsole ^
    --onefile ^
    main.py

if %ERRORLEVEL% NEQ 0 goto :error

:: Also build a normal folder version as an alternative
echo.
echo Step 5: Building a folder-based version as a backup option...
pyinstaller --noconfirm --clean ^
    --name=sagax1 ^
    --icon=assets\icons\sagax1-logo.ico ^
    --add-data="assets;assets" ^
    --add-data="config;config" ^
    --hidden-import=smolagents ^
    --hidden-import=huggingface_hub ^
    --hidden-import=transformers ^
    --hidden-import=torch ^
    --hidden-import=app ^
    --hidden-import=app.ui ^
    --hidden-import=app.core ^
    --hidden-import=app.utils ^
    --hidden-import=app.agents ^
    --hidden-import=selenium ^
    --hidden-import=helium ^
    --hidden-import=webdriver_manager ^
    --noconsole ^
    main.py

if %ERRORLEVEL% NEQ 0 goto :error

:: Step 6: Create a proper portable package
echo.
echo Step 6: Creating portable ZIP package...
cd dist

:: Create the portable package folder
if exist sagax1_Package rmdir /S /Q sagax1_Package
mkdir sagax1_Package

:: Copy the single-file executable
copy sagax1_Portable.exe sagax1_Package\

:: Create a README text file in the package
echo sagax1 Portable Version > sagax1_Package\README.txt
echo ======================= >> sagax1_Package\README.txt
echo. >> sagax1_Package\README.txt
echo This is the portable version of sagax1. >> sagax1_Package\README.txt
echo. >> sagax1_Package\README.txt
echo If the single-file executable doesn't work, try the alternative folder-based version: >> sagax1_Package\README.txt
echo 1. Extract sagax1_Folder.zip >> sagax1_Package\README.txt
echo 2. Run sagax1.exe from the extracted folder >> sagax1_Package\README.txt

:: Create a folder version backup
if exist sagax1_Folder.zip del sagax1_Folder.zip
powershell Compress-Archive -Path sagax1 -DestinationPath sagax1_Package\sagax1_Folder.zip

:: Create the final portable ZIP with both options
if exist sagax1_Portable.zip del sagax1_Portable.zip
powershell Compress-Archive -Path sagax1_Package -DestinationPath sagax1_Portable.zip

:: Clean up
rmdir /S /Q sagax1_Package

cd ..

:: Success message
echo.
echo ================================================
echo Packaging completed successfully!
echo.
echo You can find the portable executable at:
echo   dist\sagax1_Portable.exe
echo.
echo You can find the folder-based executable at:
echo   dist\sagax1\sagax1.exe
echo.
echo A portable ZIP with both options has been created at:
echo   dist\sagax1_Portable.zip
echo ================================================
goto :end

:error
echo.
echo ================================================
echo Packaging failed!
echo ================================================

:end
:: Deactivate virtual environment
call venv\Scripts\deactivate.bat
echo.
echo Press any key to exit...
pause > nul