"""Atomically publish a sibling directory without replacing any destination."""

import ctypes
import errno
import os
import stat
import sys
from pathlib import Path


AT_FDCWD = -100
RENAME_NOREPLACE = 1
RENAME_EXCL = 0x4


def _native_rename_noreplace(source, destination):
    """Invoke the platform's atomic no-replace directory rename primitive."""
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if b"\0" in source_bytes or b"\0" in destination_bytes:
        raise ValueError("publish paths cannot contain NUL bytes")
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(library, "renamex_np", None)
        arguments = (source_bytes, destination_bytes, RENAME_EXCL)
        argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        arguments = (AT_FDCWD, source_bytes, AT_FDCWD, destination_bytes,
                     RENAME_NOREPLACE)
        argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                    ctypes.c_uint)
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory publication is unsupported")
    if function is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename primitive is unavailable")
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(*arguments)
    failure = ctypes.get_errno()
    if result == 0:
        return
    if failure in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(failure, "refusing to overwrite immutable directory", destination)
    raise OSError(failure or errno.EIO, "atomic no-replace directory publication failed",
                  destination)


def publish_directory_noreplace(source, destination):
    """Move one complete sibling directory into place, failing if the target exists."""
    source = Path(source)
    destination = Path(destination)
    try:
        source_stat = source.lstat()
    except OSError as error:
        raise ValueError("publication source must be an existing real directory") from error
    if not stat.S_ISDIR(source_stat.st_mode) or source.is_symlink():
        raise ValueError("publication source must be an existing real directory")
    parent = destination.parent
    try:
        parent_stat = parent.stat()
    except OSError as error:
        raise ValueError("publication destination parent must exist") from error
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("publication destination parent must be a directory")
    if source_stat.st_dev != parent_stat.st_dev:
        raise OSError(errno.EXDEV, "publication source and destination must share a filesystem")
    if os.path.abspath(source) == os.path.abspath(destination):
        raise ValueError("publication source and destination must differ")
    _native_rename_noreplace(source, destination)
