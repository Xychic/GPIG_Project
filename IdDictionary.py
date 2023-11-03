import random
import string
class IdGenerator:
    def __init__(self,length =2,alphabet=string.ascii_letters):
        self.length = length
        self.alphabet = alphabet
        self.inUse = set()
    def gen_id(self):
        new_id = ''.join(random.choice(self.alphabet) for i in range(self.length))
        while new_id in self.inUse:
             new_id = ''.join(random.choice(self.alphabet) for i in range(self.length))
        return new_id


class IdDict:
    """
    A dictionary that gives each object added a  unique integer ID
    Can be used to get either a Object or its ID with either.
    """
    def __init__(self) -> None:
        self._data = {}
        self._counter =0

    def add(self,object):
        id = self._counter
        self.counter +=1
        self._data[id] = object

        return id
    def get_obj(self,id):
        return self._data.get(id)
    
    def get_id(self,object):
        return self._data.keys()[self._data.values.index(object)]
    
    def del_by_id(self,id):
        del self._data[id]
    
    def del_by_obj(self,object):
        id = self.get_id(object)
        del self.data[id]

    def get_ids(self):
        return list(self.data.keys())
    
    def get_objs(self):
        return list(self.data.values())
    
    def __len__(self):
        return len(self._data)

    def __str__(self) -> str:
        return str(self._data)
    