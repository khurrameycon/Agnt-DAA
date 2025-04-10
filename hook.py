
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
