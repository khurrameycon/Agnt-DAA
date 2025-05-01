
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
