
# hooks/hook-langchain_huggingface.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('langchain_huggingface')
hiddenimports.extend(['langchain_huggingface.embeddings', 'langchain_huggingface.llms'])
