"""Import-only shim for MMDetection modules unused by CLRerNet.

MMDetection 3.3 imports its complete model catalogue eagerly.  That catalogue
checks for many compiled ``mmcv._ext`` symbols even though CLRerNet calls none
of them (it has a separate, parity-tested lane NMS extension).  The current
RTX-5080 environment uses Torch 2.11, while upstream MMCV 2.1's unrelated ops
no longer compile against that C++ API.

This module lets those eager imports complete but fails closed if any compiled
MMCV operation is actually invoked.  It therefore cannot silently substitute
different numerical behavior.
"""

from __future__ import annotations

from typing import Any, Callable


def __getattr__(name: str) -> Callable[..., Any]:
    def unavailable(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(
            "CLRerNet import-only mmcv._ext shim was called for operation "
            f"{name!r}; aborting rather than changing baseline semantics"
        )

    unavailable.__name__ = name
    unavailable.__qualname__ = name
    return unavailable

