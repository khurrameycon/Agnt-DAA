# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None

# Define all packages that need complete collection
packages_to_collect = [
    'PyQt6', 'torch', 'transformers', 'huggingface_hub', 'selenium', 
    'webdriver_manager', 'smolagents', 'gradio_client', 'httpx', 'certifi', 'duckduckgo-search'
]

# Add data files for packages that need them
datas = []
for package in packages_to_collect:
    try:
        package_datas = collect_data_files(package)
        datas.extend(package_datas)
    except Exception as e:
        print(f"Warning: Could not collect data files for {package}: {e}")

# Add metadata for certificate handling and other important packages
metadata_packages = ['certifi', 'gradio_client', 'httpx']
for package in metadata_packages:
    try:
        metadata = copy_metadata(package)
        datas.extend(metadata)
    except Exception as e:
        print(f"Warning: Could not collect metadata for {package}: {e}")

# Add application-specific files and directories
datas.extend([
    ('app/', 'app/'),
    ('config/', 'config/'),
    ('assets/', 'assets/') if os.path.exists('assets') else ('', '')
])

# Create necessary directories if they don't exist
for dir_name in ['logs', 'fine_tuned_models', 'temp']:
    os.makedirs(dir_name, exist_ok=True)
    datas.append((dir_name, dir_name))

# Collect hidden imports - add all gradio client related modules
hidden_imports = [
    'torch', 'transformers', 'peft', 'huggingface_hub', 'datasets',
    'smolagents', 'selenium', 'helium', 'PIL', 'pillow', 'bs4', 'pandas', 'numpy',
    'matplotlib', 'webdriver_manager.chrome', 'webdriver_manager.core',
    'dotenv', 'logging', 'json', 'requests', 'tqdm', 
    'gradio_client', 'gradio_client.client', 'gradio_client.serializing',
    'httpx', 'httpx._config', 'httpx._client', 'httpx._models', 
    'certifi', 'ssl', 'websockets', 'websockets.client',
    'huggingface_hub.file_download', 'huggingface_hub.utils',
    'transformers.utils', 'transformers.models', 'app.agents.agent_registry',
    'app.agents.web_browsing_agent', 'app.agents.visual_web_agent',
    'app.agents.code_gen_agent', 'app.agents.local_model_agent',
    'app.agents.media_generation_agent', 'app.agents.fine_tuning_agent',
    'app.utils.style_system', 'app.utils.ui_assets', 'app.utils.logging_utils'
]

# Add gradio client submodules
hidden_imports.extend(collect_submodules('gradio_client'))
hidden_imports.extend(collect_submodules('httpx'))

# Create a runtime hook to set up certificates and temporary directory
runtime_hooks = []
with open('hook.py', 'w') as f:
    f.write('''
import os
import sys
import tempfile
import certifi
import ssl

def setup_app_environment():
    # Set SSL certificate path
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    
    # Create and set a temp directory that definitely exists and is writable
    temp_dir = os.path.join(os.path.dirname(sys.executable), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    os.environ['TMPDIR'] = temp_dir
    os.environ['TEMP'] = temp_dir
    os.environ['TMP'] = temp_dir
    tempfile.tempdir = temp_dir

setup_app_environment()
''')
runtime_hooks.append('hook.py')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
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
    console=True,  # Keep as True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='sagax1_app',  # Use a different name to avoid permission issues
)