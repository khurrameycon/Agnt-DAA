# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Add data files - important for including config files, assets, etc.
added_files = [
    ('assets', 'assets'),
    ('config', 'config'),
    ('.env', '.'),  # Include .env file if it exists
]

# Add all packages that need to be included completely
hidden_imports = [
    'smolagents',
    'huggingface_hub',
    'torch',
    'transformers',
    'app',
    'app.ui',
    'app.core',
    'app.utils',
    'app.agents',
    'selenium',
    'helium',
    'webdriver_manager',
    'webdriver_manager.chrome',
] + collect_submodules('app')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sagax1',  # Renamed from 'main' to 'sagax1'
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Changed to False for a windowed application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icons/sagax1-logo.ico',  # Add an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='sagax1',  # Renamed from 'main' to 'sagax1'
)

# For Windows, create an installer
import sys
if sys.platform == 'win32':
    from PyInstaller.utils.win32.versioninfo import VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct
    
    version_info = VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=(0, 1, 0, 0),
            prodvers=(0, 1, 0, 0),
            mask=0x3f,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0)
        ),
        kids=[
            StringFileInfo([
                StringTable(
                    '040904B0',
                    [
                        StringStruct('CompanyName', 'sagax1'),
                        StringStruct('FileDescription', 'sagax1 - Opensource AI-powered agent platform'),
                        StringStruct('FileVersion', '0.1.0'),
                        StringStruct('InternalName', 'sagax1'),
                        StringStruct('LegalCopyright', 'Copyright (c) 2025 sagax1'),
                        StringStruct('OriginalFilename', 'sagax1.exe'),
                        StringStruct('ProductName', 'sagax1'),
                        StringStruct('ProductVersion', '0.1.0'),
                    ]
                )
            ]),
            VarFileInfo([VarStruct('Translation', [1033, 1200])])
        ]
    )

    # Create the NSIS installer using the NSIS option in PyInstaller
    # Note: This requires NSIS to be installed on your system
    NSIS_OPTIONS = '''
    !define APP_NAME "sagax1"
    !define COMP_NAME "sagax1"
    !define VERSION "0.1.0"
    !define COPYRIGHT "Copyright (c) 2025 sagax1"
    !define DESCRIPTION "Opensource AI-powered agent platform"
    !define LICENSE_TXT "LICENSE.txt"
    !define INSTALLER_NAME "sagax1_Setup.exe"
    !define MAIN_APP_EXE "sagax1.exe"
    !define INSTALL_TYPE "SetShellVarContext current"
    !define REG_ROOT "HKCU"
    !define REG_APP_PATH "Software\\${APP_NAME}"
    !define UNINSTALL_PATH "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}"

    # Modern interface settings
    !include "MUI2.nsh"
    !define MUI_ABORTWARNING
    !define MUI_ICON "assets\\icons\\sagax1-logo.ico"
    !define MUI_UNICON "assets\\icons\\sagax1-logo.ico"

    # Pages
    !insertmacro MUI_PAGE_WELCOME
    !insertmacro MUI_PAGE_DIRECTORY
    !insertmacro MUI_PAGE_INSTFILES
    !insertmacro MUI_PAGE_FINISH

    # Uninstaller pages
    !insertmacro MUI_UNPAGE_CONFIRM
    !insertmacro MUI_UNPAGE_INSTFILES

    # Languages
    !insertmacro MUI_LANGUAGE "English"

    # Create desktop shortcut
    CreateShortCut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\${MAIN_APP_EXE}"
    
    # Create start menu shortcut
    CreateDirectory "$SMPROGRAMS\\${APP_NAME}"
    CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk" "$INSTDIR\\${MAIN_APP_EXE}"
    CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\Uninstall.lnk" "$INSTDIR\\uninstall.exe"
    
    # Write uninstaller
    WriteUninstaller "$INSTDIR\\uninstall.exe"
    
    # Write registry keys for uninstall
    WriteRegStr ${REG_ROOT} "${REG_APP_PATH}" "" "$INSTDIR"
    WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}" "DisplayName" "${APP_NAME}"
    WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}" "UninstallString" "$INSTDIR\\uninstall.exe"
    WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}" "DisplayIcon" "$INSTDIR\\${MAIN_APP_EXE}"
    WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}" "DisplayVersion" "${VERSION}"
    WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}" "Publisher" "${COMP_NAME}"
    '''