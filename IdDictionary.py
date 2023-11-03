import random
import string

class IdGenerator:
    def __init__(self,length =2,alphabet=string.ascii_uppercase):
        self.length = length
        self.alphabet = alphabet
        self.inUse = set()
    def gen_id(self):
        new_id = ''.join(random.choice(self.alphabet) for i in range(self.length))
        while new_id in self.inUse:
             new_id = ''.join(random.choice(self.alphabet) for i in range(self.length))
        self.inUse.add(new_id)
        return new_id
    def remove_id(self,id):
        self.inUse.remove(id)

    def __len__(self):
        return len(self.inUse)
    def get_used_ids(self):
        return list(self.inUse)




class IdDict:
    """
    A dictionary that gives each object added a  unique integer ID
    Can be used to get either a Object or its ID with either.
    Must make use of the IdGenerator if more than one is used.
    """
    def __init__(self,uniqueId="") -> None:
        self._data = {}
        self._counter =0
        self.uniqueId = uniqueId#used to differentiate the ids when multiple instances of this class exist.


    def add(self,object):
        id = self.uniqueId + "_" + str(self._counter)
        self._counter +=1
        self._data[id] = object

        return id
    def get_obj(self,id):
        return self._data.get(id)
    
    def get_id(self,object):
        keys = list(self._data.keys())
        values = list(self._data.values())
        return(keys[values.index(object)])
    
    def del_by_id(self,id):
        del self._data[id]
    
    def del_by_obj(self,object):
        id = self.get_id(object)
        del self._data[id]

    def get_ids(self):
        return list(self._data.keys())
    
    def get_objs(self):
        return list(self._data.values())
    
    def __len__(self):
        return len(self._data)

    def __str__(self) -> str:
        return str(self._data)
    