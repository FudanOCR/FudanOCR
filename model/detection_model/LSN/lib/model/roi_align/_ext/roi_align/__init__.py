
from ._roi_align import lib as _lib, ffi as _ffi

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())
