
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
