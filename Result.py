from typing import Any, Generic, Literal, NoReturn, TypeVar, TypeAlias

T = TypeVar("T")

class Ok(Generic[T]):
    def __init__(self, value: T) -> None:
        self._value: T = value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Ok) and self._value == other._value

    def __ne__(self, other: Any) -> bool:
        return self != other

    def is_ok(self) -> Literal[True]:
        return True

    def is_err(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return f"Ok({self._value})"

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value


E = TypeVar("E")
class Err(Generic[E]):
    def __init__(self, value: E) -> None:
        self._value: E = value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Err) and self._value == other._value

    def __ne__(self, other: Any) -> bool:
        return self != other

    def is_ok(self) -> Literal[False]:
        return False

    def is_err(self) -> Literal[True]:
        return True

    def __repr__(self) -> str:
        return f"Err({self._value})"

    def unwrap(self) -> NoReturn:
        raise Exception(f"Unwrap error: called unwrap on `Err` with value: {self._value}")

    def unwrap_or(self, default: T) -> T:
        return default

Result: TypeAlias = Ok[T] | Err[E]
