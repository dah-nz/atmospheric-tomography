import os
import pickle

class Cache():
    """
    Class that has cached data

    Attributes
    ----------
    name            Name of cache data directory

    Methods
    -------
    check       Checks cache for existing data
    add         Adds new data to cache
    """
    def __init__(self, name, max_len: int=256):
        self.name = name
        self._open_cache_file = None
        self._open_cache_fileptr = None
        self._current_cache = {}
        self._cache_len = 0
        self._max_length = max_len
        self._counter = 0
        self._save_frequency = 10

    def __hashed_key(self, key, hash_key=None):
        if hash_key is not None:
            return hash(hash_key) * self._max_length + key
        else:
            return key


    def check(self, key: int, hash_key=None):
        """
        Check if cache has data for a key value

        Parameters
        ----------
        key: int
            Checks cache for key

        Output
        ------
        data:
            Data associated with key. None if key is not present

        """
        key = self.__hashed_key(key, hash_key=hash_key)
        if key is not None:
            try:
                key_range_min = key - (key % self._max_length)
                key_range_max = key_range_min + self._max_length - 1
                d = os.getcwd()
                cache_file_name = "{}_{}_{}.data".format(self.name, key_range_min, key_range_max)
                cache_file_path = os.path.join(d, 'data', self.name)
                if not os.path.exists(cache_file_path):
                    os.mkdir(cache_file_path)
                cache_file = os.path.join(cache_file_path, cache_file_name)
                if cache_file != self._open_cache_file or self._counter >= self._save_frequency:
                    self.__save()
                    self.__load(cache_file)
                    self._counter += 1
                    # print(self._counter)
                    if self._counter >= self._save_frequency:
                        self._counter = 0
                return self._current_cache[key]
            except KeyError:
                pass
            except EOFError:
                pass
            except FileNotFoundError:
                self._open_cache_fileptr = open(cache_file, "wb")
                self._open_cache_file = cache_file
                self._current_cache = {}
        
        return None
    
    def add(self, key: int, data, hash_key=None):
        """
        Add data to a cache

        Parameters
        ----------
        key:
            Key to associate with data
        data:
            Data to place in cache

        Raises
        ------
        KeyError
            If the key already exists and has different data to the new data
        """
        d = self.check(key, hash_key=hash_key)
        if d is not None and hash(d) != hash(data):
            raise KeyError("key {} already exists with different data in cache".format(key))
        self._current_cache[key] = data

    def force_add(self, key: int, data, hash_key=None):
        """
        Force add data to a cache, overwriting conflicts

        Parameters
        ----------
        key:
            Key to associate with data
        data:
            Data to place in cache
        """
        key = self.__hashed_key(key, hash_key=hash_key)
        self._current_cache[key] = data

    def __save(self):
        if self._open_cache_fileptr is not None:
            self._open_cache_fileptr.close()
            if self._cache_len != len(self._current_cache.items()):
                self._open_cache_fileptr = open(self._open_cache_file, 'wb')
                pickle.dump(self._current_cache, self._open_cache_fileptr)
                self._open_cache_fileptr.close()

    def __load(self, cache_file):
        self._open_cache_fileptr = open(cache_file, "rb")
        self._open_cache_file = cache_file
        self._current_cache = pickle.load(self._open_cache_fileptr)
        self._cache_len = len(self._current_cache.items())