# hooks/hook-gradio_client.py
# from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# # Collect all data files
# datas = collect_data_files('gradio_client')

# # Add the types.json file specifically
# datas.append(('gradio_client/types.json', 'gradio_client'))

# # Get all hidden imports
# hiddenimports = collect_submodules('gradio_client')


# hooks/hook-duckduckgo_search.py
from PyInstaller.utils.hooks import collect_all

# Collect all packages, data files, and binaries
datas, binaries, hiddenimports = collect_all('gradio_client')