# Save this script as check_nsis.py
import os
import sys
import subprocess
import urllib.request
import tempfile
import zipfile
import shutil
from pathlib import Path

def is_nsis_installed():
    """Check if NSIS is installed on the system"""
    try:
        # Try to run makensis to check if it's in the PATH
        result = subprocess.run(['makensis', '/VERSION'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def download_nsis():
    """Download portable NSIS for Windows"""
    print("NSIS not found. Downloading portable version...")
    nsis_url = "https://sourceforge.net/projects/nsis/files/NSIS%203/3.08/nsis-3.08.zip/download"
    
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "nsis.zip")
        
        # Download NSIS zip file
        print("Downloading NSIS...")
        urllib.request.urlretrieve(nsis_url, zip_path)
        
        # Extract to temporary directory
        print("Extracting NSIS...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the nsis directory in the extracted files
        for item in os.listdir(temp_dir):
            if item.startswith("nsis-"):
                nsis_dir = os.path.join(temp_dir, item)
                break
        else:
            raise Exception("Could not find NSIS directory in the extracted files")
        
        # Copy to the tools directory
        tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
        os.makedirs(tools_dir, exist_ok=True)
        
        nsis_dest = os.path.join(tools_dir, "nsis")
        if os.path.exists(nsis_dest):
            shutil.rmtree(nsis_dest)
        
        shutil.copytree(nsis_dir, nsis_dest)
        
        # Add to PATH temporarily
        os.environ["PATH"] = os.path.join(nsis_dest, "bin") + os.pathsep + os.environ["PATH"]
        
        print(f"NSIS installed to {nsis_dest}")
        print("Added NSIS to PATH environment variable for this session")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"Error downloading NSIS: {str(e)}")
        return False

def main():
    """Check for NSIS installation and download if needed"""
    if is_nsis_installed():
        print("NSIS is already installed.")
        return True
    
    if sys.platform == 'win32':
        return download_nsis()
    else:
        print("This script only supports Windows.")
        return False

if __name__ == "__main__":
    main()