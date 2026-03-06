from enum import Enum, unique


try:
    from enum import StrEnum
except ImportError:
    # Python 3.10 lacks enum.StrEnum; keep string-enum semantics for older runtime envs.
    class StrEnum(str, Enum):
        pass


__all__ = ["StrEnum", "unique"]
