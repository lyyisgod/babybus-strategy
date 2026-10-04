"""One immutable account decision; estimates are a read-only annotation."""
from collections.abc import Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k:freeze(v) for k,v in value.items()})
    if isinstance(value, (list,tuple)):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {k:thaw(v) for k,v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class Decision:
    regime: str
    stress: Mapping
    constraints: Mapping
    action: str
    cores: tuple
    satellites: tuple
    rejects: tuple
    trace: Mapping
    size_unit: float = 0.
    surge: tuple = ()

    def __post_init__(self):
        for name in ('stress','constraints','cores','satellites','rejects','trace','surge'):
            object.__setattr__(self,name,freeze(getattr(self,name)))

    def to_dict(self):
        return {name:thaw(getattr(self,name)) for name in self.__dataclass_fields__}

    def to_json(self):
        return json.dumps(self.to_dict(),sort_keys=True,ensure_ascii=False,allow_nan=False)

    def with_surge(self, estimates):
        """Replacing displayed estimates cannot re-run or alter account rules."""
        from dataclasses import replace
        return replace(self,surge=tuple(estimates))
