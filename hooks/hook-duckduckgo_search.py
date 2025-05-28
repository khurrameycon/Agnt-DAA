# hooks/hook-duckduckgo_search.py
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# 1) Collect all Python modules, data files, and binary stubs from duckduckgo_search
datas, binaries, hiddenimports = collect_all('duckduckgo_search')

# 2) Explicitly include the compiled extension for utils_chat__mypyc
hiddenimports.append('duckduckgo_search.libs.utils_chat__mypyc')

# 3) Collect the dynamic library (.pyd/.so) files
additional_binaries = collect_dynamic_libs('duckduckgo_search')

# 4) Manually set the destination path in binaries
for src, _ in additional_binaries:
    if 'utils_chat__mypyc' in src:
        binaries.append((src, 'duckduckgo_search/libs'))