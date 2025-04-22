# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None

# Add duckduckgo_search binaries explicitly
binaries = []
try:
    import duckduckgo_search
    duckduckgo_search_path = os.path.dirname(duckduckgo_search.__file__)
    for f in os.listdir(duckduckgo_search_path):
        if f.endswith('.so') or f.endswith('.pyd'):
            full_path = os.path.join(duckduckgo_search_path, f)
            binaries.append((full_path, 'duckduckgo_search'))
    
    # Add libs directory if it exists
    libs_path = os.path.join(duckduckgo_search_path, 'libs')
    if os.path.exists(libs_path):
        for f in os.listdir(libs_path):
            if f.endswith('.so') or f.endswith('.pyd'):
                full_path = os.path.join(libs_path, f)
                binaries.append((full_path, 'duckduckgo_search/libs'))
except ImportError:
    print("WARNING: duckduckgo_search not found")

# Define all packages that need complete collection
packages_to_collect = [
    'PyQt6', 'transformers', 'huggingface_hub', 'selenium', 
    'webdriver_manager', 'smolagents', 'gradio_client', 'httpx', 'certifi', 
    'duckduckgo_search', 'markdown'
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
metadata_packages = ['certifi', 'gradio_client', 'httpx', 'requests', 'duckduckgo_search']
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
    ('assets/', 'assets/') if os.path.exists('assets') else ('assets', 'assets')
])

# Add SSL certificates from certifi
import certifi
datas.append((certifi.where(), 'certifi'))

# Hidden imports
hidden_imports = [
    # Core packages
    'transformers', 'peft', 'huggingface_hub', 'datasets',
    'smolagents', 'selenium', 'helium', 'PIL', 'pillow', 'bs4', 'pandas', 'numpy',
    'matplotlib', 'dotenv', 'logging', 'json', 'requests', 'tqdm', 'markdown',
    
    # Web-related
    'webdriver_manager.chrome', 'webdriver_manager.core',
    'selenium.webdriver.chrome.service',
    
    # Network and SSL
    'httpx', 'certifi', 'ssl', 'websockets', 'websockets.client',
    'requests.packages.urllib3.util.ssl_',
    
    # App modules and important tools
    'app.agents.agent_registry',
    'app.agents.web_browsing_agent', 
    'app.agents.visual_web_agent',
    'app.agents.code_gen_agent', 
    'app.agents.local_model_agent',
    'app.agents.media_generation_agent', 
    'app.agents.fine_tuning_agent',
    'app.agents.base_agent',
    'app.utils.style_system', 
    'app.utils.ui_assets', 
    'app.utils.logging_utils',
    
    # Explicitly include duckduckgo_search modules
    'duckduckgo_search',
    'duckduckgo_search.duckduckgo_search',
    'smolagents.default_tools',
]

# Create a file for the runtime hook
with open('runtime_hook.py', 'w') as f:
    f.write('''
import os
import sys
import tempfile
import certifi
import ssl

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Create and set a temp directory
base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
temp_dir = os.path.join(base_dir, 'temp')
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMPDIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir
tempfile.tempdir = temp_dir

# Create necessary directories
for dir_name in ['logs', 'fine_tuned_models', 'temp', 'assets/icons']:
    os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)

# Add a monkey patch for duckduckgo_search to avoid the compiled module issue
try:
    import sys
    import types
    
    # Create an empty module to replace the compiled one
    mock_module = types.ModuleType('duckduckgo_search.libs.utils_chat__mypyc')
    sys.modules['duckduckgo_search.libs.utils_chat__mypyc'] = mock_module
    
    # Add the minimal functionality needed
    def get_results_from_chat(chat, *args, **kwargs):
        return {}
    
    mock_module.get_results_from_chat = get_results_from_chat
except Exception as e:
    print(f"Failed to monkey patch duckduckgo_search: {e}")
''')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
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
    icon='assets/icons/sagax1-logo.ico' if os.path.exists('assets/icons/sagax1-logo.ico') else None,
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