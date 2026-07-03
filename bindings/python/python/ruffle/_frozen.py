"""A read-only mapping for result and view objects.

Result objects expose their per-channel maps read-only, so a caller cannot mutate a
:class:`ruffle.Fused` in place and then reason from a value the engine never
produced. ``types.MappingProxyType`` would provide the same protection but cannot be
pickled or deep-copied, which would make every result object unusable with
``pickle``, ``copy.deepcopy``, and ``multiprocessing``. ``FrozenMap`` is a ``dict``
whose mutators raise, so it keeps the full read surface (including ``|``,
iteration order, and equality with plain dicts) while remaining copyable and
picklable.
"""

from __future__ import annotations

from typing import Any, NoReturn, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")

__all__ = ["FrozenMap"]


class FrozenMap(dict[_K, _V]):
    """A ``dict`` whose mutating methods raise :class:`TypeError`.

    Reads behave exactly like ``dict``; ``==`` compares by content against any
    mapping. ``copy.copy``, ``copy.deepcopy``, and ``pickle`` all work and produce
    ``FrozenMap`` instances, while the ``.copy()`` method and the ``|`` operator
    return plain mutable ``dict``s, matching what ``types.MappingProxyType`` and
    ``dict`` do.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict.__repr__(self)})"

    def __reduce__(self) -> tuple[Any, ...]:
        # The default dict-subclass pickling repopulates through __setitem__, which
        # raises here; reconstructing through the constructor does not.
        return (type(self), (dict(self),))

    @classmethod
    def fromkeys(cls, iterable: Any, value: Any = None) -> Any:
        # dict.fromkeys populates through __setitem__, which raises here; building
        # the contents first keeps the alternate constructor working.
        return cls(dict.fromkeys(iterable, value))

    def __setitem__(self, key: _K, value: _V) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")

    def __delitem__(self, key: _K) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")

    # mypy requires __ior__ to stay signature-compatible with dict.__or__ (the
    # fallback for |=); a mutator that only raises cannot be, hence the override.
    def __ior__(self, other: Any) -> NoReturn:  # type: ignore[misc]
        raise TypeError(f"{type(self).__name__} is read-only")

    def clear(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")

    def pop(self, *args: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")

    def popitem(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")

    def setdefault(self, *args: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")
