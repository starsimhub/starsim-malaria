"""
malariasim/updaters — auto-import all updater modules.

Importing this package causes every module in the directory to be imported,
which fires each Updater subclass's ``__init_subclass__`` hook and populates
``Updater._possible_updaters``.  No list to maintain — adding a new file in
this directory is sufficient to register its updater.
"""

import importlib
import pkgutil
from pathlib import Path

for _mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
    importlib.import_module(f'.{_mod.name}', package=__name__)
