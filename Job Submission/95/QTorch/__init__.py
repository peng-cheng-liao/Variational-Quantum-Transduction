import importlib
import os
import pathlib

__all__ = []

# Get current package directory
current_dir = pathlib.Path(__file__).parent

for file in os.listdir(current_dir):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]  # strip .py
        module = importlib.import_module(f".{module_name}", package=__name__)
        for attr in dir(module):
            if not attr.startswith("_"):
                globals()[attr] = getattr(module, attr)
                __all__.append(attr)
