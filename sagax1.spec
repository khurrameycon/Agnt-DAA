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
    'duckduckgo_search', 'markdown',
    # RAG-related packages
    'langchain', 'langchain_community', 'langchain_huggingface', 'langchain_core',
    'faiss', 'unidecode', 'sklearn', 'sentence_transformers',
    # New additions for PyQt6-WebEngine
    'PyQt6.QtWebEngineWidgets', 'PyQt6.QtWebEngineCore'
]

# Add data files for packages that need them
datas = []
models_path = 'models/sentence_transformers' 
if os.path.exists(models_path):
    for root, dirs, files in os.walk(models_path):
        for file in files:
            source_path = os.path.join(root, file)
            target_path = os.path.join(os.path.relpath(root, os.getcwd()))
            datas.append((source_path, target_path))
    print(f"Added sentence transformer model files from {models_path}")

for package in packages_to_collect:
    try:
        package_datas = collect_data_files(package)
        datas.extend(package_datas)
    except Exception as e:
        print(f"Warning: Could not collect data files for {package}: {e}")

# Add metadata for certificate handling and other important packages
metadata_packages = [
    'certifi', 'gradio_client', 'httpx', 'requests', 'duckduckgo_search',
    # RAG-related packages metadata
    'faiss', 'langchain', 'langchain_community', 'langchain_huggingface', 'sklearn'
]
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

# Add new.bat file which is used for installation
datas.extend([
    ('app/utils/new.bat', 'app/utils/')
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
    'app.agents.rag_agent',  # Add RAG agent
    'app.utils.style_system', 
    'app.utils.ui_assets', 
    'app.utils.logging_utils',
    
    # Explicitly include duckduckgo_search modules
    'duckduckgo_search',
    'duckduckgo_search.duckduckgo_search',
    'smolagents.default_tools',
    
    # RAG-related imports
    'faiss',
    'unidecode',
    'langchain',
    'langchain_community',
    'langchain_huggingface',
    'langchain_core',
    'langchain.text_splitter',
    'langchain.chains',
    'langchain.chains.conversational_retrieval',
    'langchain.memory',
    'langchain_core.prompts',
    'langchain_core.documents',
    'langchain_community.document_loaders',
    'langchain_community.document_loaders.pdf',
    'langchain_community.document_loaders.text',
    'langchain_community.vectorstores',
    'langchain_community.vectorstores.faiss',
    'langchain_huggingface.embeddings',
    'langchain_huggingface.llms',
    'sklearn.feature_extraction.text',
    'sentence_transformers',
    
    # New PyQt6-WebEngine imports
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtWebEngineCore',
    'PyQt6.QtWebEngineWidgets.QWebEngineView'
]

# Add sentence transformers related imports
hidden_imports.extend([
    'sentence_transformers',
    'sentence_transformers.models',
    'sentence_transformers.util',
    'torch',
    'transformers.models.bert',
    'transformers.models.roberta',
    'transformers.models.xlm_roberta'
])

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
for dir_name in ['logs', 'fine_tuned_models', 'temp', 'assets/icons', 'faiss_indexes']:
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

# Set up WebEngine environment variables
if getattr(sys, 'frozen', False):
    try:
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        print("Successfully imported QWebEngineCore")
    except ImportError as e:
        print(f"Warning: Could not import QWebEngineCore: {e}")
''')

# Add hook for RAG-related packages
os.makedirs('hooks', exist_ok=True)

with open('hooks/hook-faiss.py', 'w') as f:
    f.write('''
# hooks/hook-faiss.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('faiss')
''')

with open('hooks/hook-langchain.py', 'w') as f:
    f.write('''
# hooks/hook-langchain.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('langchain')
hiddenimports.extend(['langchain.text_splitter', 'langchain.chains', 'langchain.memory'])
''')

with open('hooks/hook-langchain_community.py', 'w') as f:
    f.write('''
# hooks/hook-langchain_community.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('langchain_community')
hiddenimports.extend([
    'langchain_community.document_loaders',
    'langchain_community.document_loaders.pdf',
    'langchain_community.vectorstores',
    'langchain_community.vectorstores.faiss'
])
''')

with open('hooks/hook-langchain_huggingface.py', 'w') as f:
    f.write('''
# hooks/hook-langchain_huggingface.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('langchain_huggingface')
hiddenimports.extend(['langchain_huggingface.embeddings', 'langchain_huggingface.llms'])
''')

# Add hook for PyQt6-WebEngine
with open('hooks/hook-PyQt6.QtWebEngineWidgets.py', 'w') as f:
    f.write('''
# hooks/hook-PyQt6.QtWebEngineWidgets.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('PyQt6.QtWebEngineWidgets')
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