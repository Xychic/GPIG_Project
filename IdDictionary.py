import random
import string
from typing import Generic, TypeVar


class IdGenerator:
    def __init__(self, length: int = 2, alphabet: str = string.ascii_uppercase):
        self.length: int = length
        self.alphabet: str = alphabet
        self.inUse: set[str] = set()

    def gen_id(self):
        new_id = "".join(random.choice(self.alphabet) for _ in range(self.length))
        while new_id in self.inUse:
            new_id = "".join(random.choice(self.alphabet) for _ in range(self.length))
        self.inUse.add(new_id)
        return new_id

    def remove_id(self, id: str):
        self.inUse.remove(id)

    def __len__(self):
        return len(self.inUse)

    def get_used_ids(self):
        return list(self.inUse)


T = TypeVar("T")


class IdDict(Generic[T]):

    """
    A dictionary that gives each object added a  unique integer ID
    Can be used to get either a Object or its ID with either.
    Must make use of the IdGenerator if more than one is used.
    """

    def __init__(self, uniqueId: str = "") -> None:
        self._data: dict[str, T] = {}
        self._counter: int = 0
        # Used to differentiate the ids when multiple instances of this class exist.
        self.uniqueId: str = uniqueId

    def add(self, object: T) -> str:
        id = self.uniqueId + "_" + str(self._counter)
        self._counter += 1
        self._data[id] = object
        return id

    def get_obj(self, id: str) -> T:
        result = self._data.get(id)
        if result is None:
            raise Exception(f"Invalid ID: {id}")
        return result

    def get_id(self, object: T) -> str:
        result = next((k for (k, v) in self._data.items() if v == object), None)
        if result is None:
            raise Exception(f"Invalid object: {object}")
        return result

    def del_by_id(self, id: str):
        del self._data[id]

    def del_by_obj(self, object: T):
        id = self.get_id(object)
        del self._data[id]

    def get_ids(self) -> list[str]:
        return list(self._data.keys())

    def get_objs(self) -> list[T]:
        return list(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        return str(self._data)
