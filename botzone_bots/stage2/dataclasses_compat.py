"""Minimal ``dataclasses`` for Botzone's Python 3.6 interpreter.

Covers only what the bundled Stage 2 modules use: ``dataclass`` with
``frozen``/``order``, ``field(default=..., default_factory=...)``,
``fields``, ``replace`` and ``asdict``, including dataclass inheritance.
``__main__`` installs it only when the real module cannot be imported.
"""

import copy

__all__ = ["dataclass", "field", "fields", "replace", "asdict", "FrozenInstanceError", "MISSING"]


class _Missing(object):
    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


class FrozenInstanceError(AttributeError):
    pass


class Field(object):
    __slots__ = ("name", "default", "default_factory")

    def __init__(self, default=MISSING, default_factory=MISSING):
        if default is not MISSING and default_factory is not MISSING:
            raise ValueError("cannot specify both default and default_factory")
        self.name = None
        self.default = default
        self.default_factory = default_factory


def field(default=MISSING, default_factory=MISSING):
    return Field(default, default_factory)


def fields(obj):
    return tuple(type(obj).__dataclass_fields__.values() if not isinstance(obj, type)
                 else obj.__dataclass_fields__.values())


def _collect(cls):
    collected = {}
    for base in reversed(cls.__mro__[1:]):
        collected.update(getattr(base, "__dataclass_fields__", {}))
    for name in cls.__dict__.get("__annotations__", {}):
        value = cls.__dict__.get(name, MISSING)
        item = value if isinstance(value, Field) else Field(default=value)
        item.name = name
        if isinstance(value, Field):
            if value.default is MISSING:
                delattr(cls, name)
            else:
                setattr(cls, name, value.default)
        collected[name] = item
    seen_default = False
    for item in collected.values():
        has_default = item.default is not MISSING or item.default_factory is not MISSING
        if seen_default and not has_default:
            raise TypeError("non-default argument %r follows default argument" % item.name)
        seen_default = seen_default or has_default
    return collected


def _process(cls, frozen, order):
    specs = _collect(cls)
    names = tuple(specs)
    cls.__dataclass_fields__ = specs

    def __init__(self, *args, **kwargs):
        if len(args) > len(names):
            raise TypeError("%s takes %d positional arguments" % (cls.__name__, len(names)))
        values = dict(zip(names, args))
        for key, value in kwargs.items():
            if key not in specs:
                raise TypeError("%s got an unexpected keyword argument %r" % (cls.__name__, key))
            if key in values:
                raise TypeError("%s got multiple values for argument %r" % (cls.__name__, key))
            values[key] = value
        for name in names:
            if name not in values:
                spec = specs[name]
                if spec.default_factory is not MISSING:
                    values[name] = spec.default_factory()
                elif spec.default is not MISSING:
                    values[name] = spec.default
                else:
                    raise TypeError("%s missing required argument %r" % (cls.__name__, name))
            object.__setattr__(self, name, values[name])

    def _astuple(self):
        return tuple(getattr(self, name) for name in names)

    def __repr__(self):
        return "%s(%s)" % (cls.__name__, ", ".join("%s=%r" % (n, getattr(self, n)) for n in names))

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return _astuple(self) == _astuple(other)
        return NotImplemented

    cls.__init__ = __init__
    cls.__repr__ = __repr__
    cls.__eq__ = __eq__
    if frozen:
        def __setattr__(self, name, value):
            raise FrozenInstanceError("cannot assign to field %r" % name)

        def __delattr__(self, name):
            raise FrozenInstanceError("cannot delete field %r" % name)

        cls.__setattr__ = __setattr__
        cls.__delattr__ = __delattr__
        cls.__hash__ = lambda self: hash(_astuple(self))
    else:
        cls.__hash__ = None
    if order:
        def _compare(op):
            def method(self, other):
                if other.__class__ is self.__class__:
                    return op(_astuple(self), _astuple(other))
                return NotImplemented
            return method
        cls.__lt__ = _compare(lambda a, b: a < b)
        cls.__le__ = _compare(lambda a, b: a <= b)
        cls.__gt__ = _compare(lambda a, b: a > b)
        cls.__ge__ = _compare(lambda a, b: a >= b)
    return cls


def dataclass(cls=None, frozen=False, order=False, eq=True):
    if not eq:
        raise NotImplementedError("eq=False is not supported by the Botzone shim")

    def wrap(target):
        return _process(target, frozen, order)

    return wrap if cls is None else wrap(cls)


def replace(obj, **changes):
    values = {name: getattr(obj, name) for name in obj.__dataclass_fields__}
    values.update(changes)
    return obj.__class__(**values)


def _copy_value(value):
    if hasattr(type(value), "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_value(item) for item in value)
    return copy.deepcopy(value)


def asdict(obj):
    return {name: _copy_value(getattr(obj, name)) for name in obj.__dataclass_fields__}
