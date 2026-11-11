# PyInstaller Runtime Hook — preload torch BEFORE any other module touches it.
#
# Problem: If anything (e.g. torchvision meta-registrations) triggers a failed
# torch import, torch sets an internal "already failed" flag and all subsequent
# `import torch` calls raise:
#   "torch can only be initialized once per process"
#
# Solution: import torch here, at the very start of the process, so the first
# import attempt succeeds cleanly.  Subsequent imports just reuse sys.modules.

import sys
import os

# Point torch to its bundled DLLs before importing
_internal = getattr(sys, "_MEIPASS", None)
if _internal:
    # Add _internal to DLL search path so torch can find its .dll files
    os.add_dll_directory(_internal)
    _torch_lib = os.path.join(_internal, "torch", "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

try:
    import torch  # noqa: F401 — preload only, do NOT remove
    print(f"[rthook] torch {torch.__version__} preloaded OK")
except Exception as _e:
    print(f"[rthook] torch preload WARNING: {_e}")
    # Don't raise — app can still run with MediaPipe fallback
