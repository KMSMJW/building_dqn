from dataclasses import dataclass

import numpy as np


@dataclass
class BaseTypeClass:
    def __len__(self):
        return len(self.__dict__)

    def to_dict(self):
        return self.__dict__
    
    def get_fields(self):
        return self.__dict__.keys()
    
    def get_values(self):
        return self.__dict__.values()
    
    def to_numpy(self):
        return np.array(list(self.get_values()), dtype=float)