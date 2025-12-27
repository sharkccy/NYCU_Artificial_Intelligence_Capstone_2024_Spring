
import os
import importlib
# 獲取 utils 資料夾的路徑
utils_path = os.path.dirname(__file__)

# 獲取 utils 資料夾中的所有 .py 檔案（除了 __init__.py）
modules = [f[:-3] for f in os.listdir(utils_path) if f.endswith('.py') and f != '__init__.py']

# 動態匯入所有模組，並將其添加到 globals()
for module in modules:
    globals()[module] = importlib.import_module(f'.{module}', package='utils')
