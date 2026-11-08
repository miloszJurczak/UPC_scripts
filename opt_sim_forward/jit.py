"""numba switch: real `njit` if numba is installed, otherwise a no-op decorator.

The compiled-code cache lives in .numba_cache/<hash of all *.py here>/. Numba's own cache
only notices edits to the file of the cached function, not to constants it imports from
other files (e.g. the index constants in params.py) - a stale cache would silently use
old indices. Keying the folder on every source file makes any edit force a recompile.
"""
import glob
import hashlib
import os

_here = os.path.dirname(os.path.abspath(__file__))
_h = hashlib.sha1()
for _f in sorted(glob.glob(os.path.join(_here, "*.py"))):
    with open(_f, "rb") as _fh:
        _h.update(_fh.read())
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_here, ".numba_cache", _h.hexdigest()[:12]))

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                   # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f
