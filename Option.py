from dataclasses import dataclass
from typing import Any, Generic, Literal, NoReturn, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Some(Generic[T]):
    _value: T

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Some):
            x: Some[T] = other
            return self._value == x._value
        return False

    def __ne__(self, other: Any) -> bool:
        return self != other

    def is_some(self) -> Literal[True]:
        return True

    def is_null(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return f"Some({self._value})"

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value


@dataclass
class NullClass:
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NullClass)

    def __ne__(self, other: Any) -> bool:
        return self != other

    def is_some(self) -> Literal[False]:
        return False

    def is_null(self) -> Literal[True]:
        return True

    def __repr__(self) -> str:
        return f"Null"

    def unwrap(self) -> NoReturn:
        raise Exception(f"Unwrap error: called unwrap on Null")

    def unwrap_or(self, default: T) -> T:
        return default


Null = NullClass()
Option = Some[T] | NullClass
