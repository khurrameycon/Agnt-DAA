# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Define all packages that need complete collection
packages_to_collect = [
    'PyQt6', 'torch', 'transformers', 'huggingface_hub', 'selenium', 
    'webdriver_manager', 'smolagents', 'gradio_client', 'gradio'
]

# Add data files for packages that need them
datas = []
for package in packages_to_collect:
    try:
        package_datas = collect_data_files(package)
        datas.extend(package_datas)
    except Exception as e:
        print(f"Warning: Could not collect data files for {package}: {e}")

# Add application-specific files and directories
datas.extend([
    ('app/', 'app/'),
    ('config/', 'config/'),
    ('assets/', 'assets/') if os.path.exists('assets') else ('', '')
])

# Create necessary directories if they don't exist
os.makedirs('logs', exist_ok=True)

# Collect hidden imports
hidden_imports = [
    'torch', 'transformers', 'peft', 'huggingface_hub', 'datasets',
    'smolagents', 'selenium', 'helium', 'PIL', 'pillow', 'bs4', 'pandas', 'numpy',
    'matplotlib', 'webdriver_manager.chrome', 'webdriver_manager.core',
    'dotenv', 'logging', 'json', 'requests', 'tqdm', 'gradio_client',
    'huggingface_hub.file_download', 'huggingface_hub.utils',
    'transformers.utils', 'transformers.models', 'app.agents.agent_registry',
    'app.agents.web_browsing_agent', 'app.agents.visual_web_agent',
    'app.agents.code_gen_agent', 'app.agents.local_model_agent',
    'app.agents.media_generation_agent', 'app.agents.fine_tuning_agent',
    'app.utils.style_system', 'app.utils.ui_assets', 'app.utils.logging_utils'
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
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
    name='sagax1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to True for debugging, change to False for final version
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Set this to your icon path if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='sagax1',
)