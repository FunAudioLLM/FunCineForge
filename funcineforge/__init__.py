"""Initialize package."""

import os

dirname = os.path.dirname(__file__)

import importlib
import pkgutil


def import_submodules(package, recursive=True):
    if isinstance(package, str):
        try:
            package = importlib.import_module(package)
        except Exception as e:
            pass
    results = {}
    if not isinstance(package, str):
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                results[name] = importlib.import_module(name)
            except Exception as e:
                import traceback
                print(f"\n❌ Import Failed: {name}")
                print(f"Error: {e}")
                traceback.print_exc()
                pass
            if recursive and is_pkg:
                results.update(import_submodules(name))
    return results

import_submodules(__name__)

from funcineforge.auto.auto_model import AutoModel
# from funcineforge.auto.auto_frontend import AutoFrontend

os.environ["HYDRA_FULL_ERROR"] = "1"
