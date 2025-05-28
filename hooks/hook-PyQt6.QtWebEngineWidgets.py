
# hooks/hook-PyQt6.QtWebEngineWidgets.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('PyQt6.QtWebEngineWidgets')
