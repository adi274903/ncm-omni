"""omni - next-concept multimodal model.

Import shim, first line of defence: keep TensorFlow out of the process.

torch imports `opt_einsum`, whose `backends/__init__` eagerly imports every backend, including
`backends/tensorflow.py` (`import tensorflow as tf` inside a try/except ImportError). On Colab TF
is installed, so that import succeeds: two `absl` initialisations, ~600 MB RSS, ~3 s of startup,
and on a healthy GPU TF claims VRAM on import. Nothing here uses TF.

Poisoning the name with None makes `import tensorflow` raise ImportError, which is exactly the
branch opt_einsum already handles, so its TF backend degrades to `tf = None`.

    NCM_ALLOW_TF=1   keep TensorFlow importable (debugging a TF path)
"""

import os as _os
import sys as _sys

if _os.environ.get("NCM_ALLOW_TF", "") != "1":
    # setdefault, not assignment: if the host already imported TF, leave it alone.
    # None is the documented way to poison an import; pyright types the mapping as ModuleType.
    _sys.modules.setdefault("tensorflow", None)          # type: ignore[arg-type]
