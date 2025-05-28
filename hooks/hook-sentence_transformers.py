# hooks/hook-sentence_transformers.py
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('sentence_transformers')

# Add specific submodules
hiddenimports.extend([
    'sentence_transformers.models',
    'sentence_transformers.models.Transformer',
    'sentence_transformers.models.Pooling',
    'sentence_transformers.models.Normalize',
    'sentence_transformers.cross_encoder',
    'sentence_transformers.SentenceTransformer'
])