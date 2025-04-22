
import os
import sys
import tempfile
import certifi
import ssl

def ensure_safehttpx_version():
    """Create safehttpx version.txt file if it doesn't exist"""
    try:
        import os
        import sys
        
        # Path where the file should be in the frozen app
        app_dir = os.path.dirname(sys.executable)
        safehttpx_dir = os.path.join(app_dir, '_internal', 'safehttpx')
        version_file = os.path.join(safehttpx_dir, 'version.txt')
        
        # Create directory if it doesn't exist
        if not os.path.exists(safehttpx_dir):
            os.makedirs(safehttpx_dir, exist_ok=True)
            
        # Create version file if it doesn't exist
        if not os.path.exists(version_file):
            with open(version_file, 'w') as f:
                f.write("1.0.0")  # Use a safe default version
            print(f"Created missing safehttpx version.txt file at {version_file}")
    except Exception as e:
        print(f"Warning: Could not create safehttpx version.txt file: {e}")



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
    ensure_safehttpx_version()

setup_app_environment()
