
# hooks/hook-langchain.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('langchain')
hiddenimports.extend(['langchain.text_splitter', 'langchain.chains', 'langchain.memory'])
