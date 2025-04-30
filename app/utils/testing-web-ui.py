import subprocess
import os
import sys

# --- Configuration ---
# !! Adjust this path if your web-ui is installed elsewhere !!
WEBUI_DIRECTORY = r"D:\web-ui"
# -------------------

# Construct the full paths
venv_python_executable = os.path.join(WEBUI_DIRECTORY, '.venv', 'Scripts', 'python.exe')
webui_script = os.path.join(WEBUI_DIRECTORY, 'webui.py')

print(f"Attempting to launch WebUI from: {WEBUI_DIRECTORY}")
print(f"Using Python executable: {venv_python_executable}")
print(f"Running script: {webui_script}")

# Check if the virtual environment Python and the script exist
if not os.path.exists(venv_python_executable):
    print(f"\nError: Python executable not found at '{venv_python_executable}'")
    print("Please ensure the virtual environment exists and the path is correct.")
    input("Press Enter to exit...")
    sys.exit(1)

if not os.path.exists(webui_script):
    print(f"\nError: WebUI script not found at '{webui_script}'")
    print("Please ensure the web-ui installation is correct.")
    input("Press Enter to exit...")
    sys.exit(1)

# Prepare the command to run
# We run the python executable from the venv and tell it to run the webui.py script.
# We set the 'cwd' (current working directory) to the WEBUI_DIRECTORY,
# which is crucial for webui.py to find its files (like .env).
command = [venv_python_executable, webui_script]

try:
    print("\nLaunching WebUI...")
    # Use Popen to start the process and let this script exit
    # The WebUI server will continue running in its own process window.
    # Use creationflags=subprocess.CREATE_NEW_CONSOLE on Windows if you
    # want the webui server to run in its own new command prompt window.
    subprocess.Popen(command, cwd=WEBUI_DIRECTORY, creationflags=subprocess.CREATE_NEW_CONSOLE)
    print("Launch command sent. The WebUI should start in a new window.")
    print("You can close this launcher window.")

except Exception as e:
    print(f"\nAn error occurred while trying to launch the WebUI: {e}")

# Keep the launcher window open briefly to show messages if run by double-click
# Remove or comment out the input() line if you run this from an existing cmd/terminal
print("\n---")
input("Press Enter to exit this launcher...")