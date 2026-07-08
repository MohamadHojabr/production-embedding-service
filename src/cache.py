from collections import OrderedDict
import numpy as np


class EmbeddingCache:

    def __init__(
        self,
        max_size: int = 10000
    ):

        self.max_size = max_size

        self.cache = OrderedDict()



    def get(
        self,
        key: str
    ):

        if key not in self.cache:
            return None


        # Move to end (recently used)
        value = self.cache.pop(key)

        self.cache[key] = value


        return value



    def set(
        self,
        key: str,
        value: np.ndarray
    ):

        if key in self.cache:
            self.cache.pop(key)


        self.cache[key] = value


        if len(self.cache) > self.max_size:

            # Remove oldest item
            self.cache.popitem(
                last=False
            )



    def clear(self):

        self.cache.clear()



    def size(self):

        return len(self.cache)